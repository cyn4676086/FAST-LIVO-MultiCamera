float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return 0.;
    StatesGroup old_state = (*state);
    V2D pc; 
    MD(1,2) Jimg;
    MD(2,3) Jdpi;
    MD(1,3) Jdphi, Jdp, JdR, Jdt;
    VectorXd z;
    // VectorXd R;
    bool EKF_end = false;
    /* Compute J */
    float error=0.0, last_error=total_residual, patch_error=0.0, last_patch_error=0.0, propa_error=0.0;
    // MatrixXd H;
    bool z_init = true;
    const int H_DIM = total_points * patch_size_total;
    
    // K.resize(H_DIM, H_DIM);
    z.resize(H_DIM);
    z.setZero();
    // R.resize(H_DIM);
    // R.setZero();

    // H.resize(H_DIM, DIM_STATE);
    // H.setZero();
    H_sub.resize(H_DIM, 6);
    H_sub.setZero();
    
    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++) 
    {
        // double t1 = omp_get_wtime();
        double count_outlier = 0;
     
        error = 0.0;
        propa_error = 0.0;
        n_meas_ =0;
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();
        
        M3D p_hat;
        int i;

        for (i=0; i<sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0;
            int search_level = sub_sparse_map->search_levels[i];
            int pyramid_level = level + search_level;
            const int scale =  (1<<pyramid_level);
            
            PointPtr pt = sub_sparse_map->voxel_points[i];

            if(pt==nullptr) continue;

            V3D pf = Rcw * pt->pos_ + Pcw;
            pc = cam->world2cam(pf);
            // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
            {
                dpi(pf, Jdpi, cam.fx, cam.fy);
                p_hat << SKEW_SYM_MATRX(pf);
            }
            const float u_ref = pc[0];
            const float v_ref = pc[1];
            const int u_ref_i = floorf(pc[0]/scale)*scale; 
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            const float subpix_u_ref = (u_ref-u_ref_i)/scale;
            const float subpix_v_ref = (v_ref-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            
            vector<float> P = sub_sparse_map->patch[i];
            for (int x=0; x<patch_size; x++) 
            {
                uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i+x*scale-patch_size_half*scale)*width + u_ref_i-patch_size_half*scale;
                for (int y=0; y<patch_size; ++y, img_ptr+=scale) 
                {
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //{
                    float du = 0.5f * ((w_ref_tl*img_ptr[scale] + w_ref_tr*img_ptr[scale*2] + w_ref_bl*img_ptr[scale*width+scale] + w_ref_br*img_ptr[scale*width+scale*2])
                                -(w_ref_tl*img_ptr[-scale] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[scale*width-scale] + w_ref_br*img_ptr[scale*width]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr[scale*width] + w_ref_tr*img_ptr[scale+scale*width] + w_ref_bl*img_ptr[width*scale*2] + w_ref_br*img_ptr[width*scale*2+scale])
                                -(w_ref_tl*img_ptr[-scale*width] + w_ref_tr*img_ptr[-scale*width+scale] + w_ref_bl*img_ptr[0] + w_ref_br*img_ptr[scale]));
                    Jimg << du, dv;
                    Jimg = Jimg * (1.0/scale);
                    Jdphi = Jimg * Jdpi * p_hat;
                    Jdp = -Jimg * Jdpi;
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                    Jdt = Jdp * Jdp_dt;
                    //}
                    double res = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale]  - P[patch_size_total*level + x*patch_size+y];
                    z(i*patch_size_total+x*patch_size+y) = res;
                    // float weight = 1.0;
                    // if(iteration > 0)
                    //     weight = weight_function_->value(res/weight_scale_); 
                    // R(i*patch_size_total+x*patch_size+y) = weight;       
                    patch_error +=  res*res;
                    n_meas_++;
                    // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR, Jdt;
                }
            }  

            sub_sparse_map->errors[i] = patch_error;
            error += patch_error;
        }

        // computeH += omp_get_wtime() - t1;

        error = error/n_meas_;

        // double t3 = omp_get_wtime();

        if (error <= last_error) 
        {
            old_state = (*state);
            last_error = error;

            // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
            // auto vec = (*state_propagat) - (*state);
            // G = K*H;
            // (*state) += (-K*z + vec - G*vec);

            auto &&H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            auto solution = - K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
            (*state) += solution;
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);

            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }

        // ekf_time += omp_get_wtime() - t3;

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
} 