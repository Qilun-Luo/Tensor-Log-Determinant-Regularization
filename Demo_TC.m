%Demo for Tensor Completion
clear
close all

rng('shuffle')

addpath(genpath('utils/'))
addpath(genpath('algs/'));
addpath(genpath('data/'));

test_result = {};
test_history = {};
test_ground_truth = {};
test_psnr = {};
test_rse = {};
test_ssim = {};
test_alg_name = {};

%% alg setting
flag_sample = 1;
% proposed
flag_logdet = 1;
flag_eps_logdet = 1;

%% --main--

p_list = 0.2:0.2:0.2;
test_list = 1:2;

for k = 1:length(p_list)
    p = p_list(k); % sample rate
    fprintf('p=%.2f\n', p_list(k))
    for test_num = test_list

        %% ----data set-------
        switch test_num
            case 1
                load mri_data.mat
                disp('MRI')
            case 2
                load msi_data.mat
                disp('MSI')
            case 3
                load video_road.mat
                disp('Road')
            case 4
                load video_basketball.mat 
                disp('Basketball')
            case 5
                load hsi_Cuprite_data.mat
                disp('HSI')
            case 6
                load salesman_data.mat
                disp('Salesman')
        end
        fprintf('Test on dataset #%d\n', test_num)
   
        %% ---------------------
        normalize = max(T(:));
        M = T/normalize;
        sz =  size(M);
    
        %% sampled tensor
        num = prod(sz);
        chosen = find(rand(num,1)<p);
        Omega = zeros(sz);
        Omega(chosen) = 1;
        D = diag(sparse(double(Omega(:)))); % sampling operator
        b = D * M(:);
        bb = reshape(b, sz);

        psnr_list = [];
        ssim_list = [];
        rse_list = [];
        runtime_list = [];
        alg_name = {};
        alg_result = {};
        alg_history = {};
        alg_cnt = 1;
        
        %% -- Sample
        if flag_sample
            tic
            X_psnr_sample = psnr(bb, M);
            X_ssim_sample = ssim(bb, M);
            X_rse_sample = norm(bb(:)-M(:))/norm(M(:));
            runtime_list(alg_cnt) = toc;
            psnr_list(alg_cnt) = X_psnr_sample;
            ssim_list(alg_cnt) = X_ssim_sample;
            rse_list(alg_cnt) = X_rse_sample;
            alg_name{alg_cnt} = 'Sample';
            alg_result{alg_cnt} = bb;
            alg_history{alg_cnt} = [];
            alg_cnt = alg_cnt + 1;
            
        end

        
        %% -- Alg: logdet --
        if flag_logdet
            opts = [];
            opts.mu = 1e-4;
            opts.tol = 1e-4;
            opts.rho = 1.2;
            opts.max_iter = 500;
            opts.DEBUG = 0;
            opts.max_mu = 1e10;
            opts.Xtrue = M;
          
            tic 
            [X_logdet, Out_logdet] = lrtc_logdet(bb, chosen, opts);
            runtime_list(alg_cnt) = toc;
            X_dif_logdet = X_logdet - M;
            res_logdet = norm(X_dif_logdet(:))/norm(M(:));
            X_psnr_logdet = psnr(X_logdet, M);
            X_ssim_logdet = ssim(X_logdet, M);
            X_rse_logdet = norm(X_logdet(:)-M(:))/norm(M(:));
            psnr_list(alg_cnt) = X_psnr_logdet;
            ssim_list(alg_cnt) = X_ssim_logdet; 
            rse_list(alg_cnt) = X_rse_logdet;
            alg_name{alg_cnt} = 'LogDet';
            alg_result{alg_cnt} = X_logdet;
            alg_history{alg_cnt} = Out_logdet;
            alg_cnt = alg_cnt + 1;
        end
        
        %% -- Alg: epsilon-logdet --
        if flag_eps_logdet
            opts = [];
            opts.mu = 1e-4;
            opts.tol = 1e-4;
            opts.rho = 1.2;
            opts.max_iter = 500;
            opts.DEBUG = 0;
            opts.max_mu = 1e10;
            opts.Xtrue = M;
            tic
            [X_logdet_2, history_logdet_2] = lrtc_epsilon_logdet(bb, chosen, opts);
            runtime_list(alg_cnt) = toc;
            X_dif_logdet_2 = X_logdet_2 - M;
            res_logdet_2 = norm(X_dif_logdet_2(:))/norm(M(:));
            X_psnr_logdet_2 = psnr(X_logdet_2, M);
            E = X_logdet_2-M;
            X_ssim_logdet_2 = ssim(X_logdet_2, M);
            X_rse_logdet_2 = norm(X_logdet_2(:)-M(:))/norm(M(:));
            psnr_list(alg_cnt) = X_psnr_logdet_2;
            ssim_list(alg_cnt) = X_ssim_logdet_2;
            rse_list(alg_cnt) = X_rse_logdet_2;
            alg_name{alg_cnt} = 'eps-LogDet';
            alg_result{alg_cnt} = X_logdet_2;
            alg_history{alg_cnt} = history_logdet_2;
            alg_cnt = alg_cnt + 1;
        end
        
        
        %% -- Result table--
        fprintf('%4s', 'metric');
        for j = 1:alg_cnt-1
            fprintf('\t%4s', alg_name{j});
        end
        fprintf('\n')

        fprintf('%4s', 'PSNR');
        for j = 1:alg_cnt-1
            fprintf('\t%.4f', psnr_list(j));
        end
        fprintf('\n')

        fprintf('%4s', 'SSIM');
        for j = 1:alg_cnt-1
            fprintf('\t%.4f', ssim_list(j));
        end
        fprintf('\n')
        
        fprintf('%4s', 'RSE');
        for j = 1:alg_cnt-1
            fprintf('\t%.4f', rse_list(j));
        end
        fprintf('\n')
        
        fprintf('%4s', 'CPU');
        for j = 1:alg_cnt-1
            fprintf('\t%.4f', runtime_list(j));
        end
        fprintf('\n')
      
 
    end

end
