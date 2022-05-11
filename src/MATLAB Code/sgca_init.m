function [sol, sol1, Sigma] = sgca_init(X, pp, r, pen, tol, maxiter)

% Inputs:
% =======
% X:       data sets, n rows correspond to samples and p columns to variables
% pp:      a length k vector giving the sizes of each variable subset
% r:       nuclear norm constraint
% pen:     penalty parameter on the l_1 norm of the solution, scaled by
%          sqrt(log(max(p1,p2))/n)
% tol:     tolerance level for convergence in ADMM
% maxiter: maximum number of iterations in ADMM
% 
% Outputs:
% ========
% sol:     optimum of the convex program
% sol1:    resacled optimum of the convex program, premultiplied by
%          SXX^(0.5), post multiplied by SYY^(0.5)

[n, p] = size(X);

% ssX = std(X);
% X = X - ones(n,1) * mean(X);
% X = X ./ (ones(n,1) * ssX);
% X = X./sqrt(n);

k = numel(pp);
pp = reshape(pp,1,[]);
pp = [0, cumsum(pp)];
Sigma = cov(X);
Mask = zeros(p,p);
for i=1:k
    idx = pp(i)+1:pp(i+1);
    Mask(idx,idx) = ones(numel(idx), numel(idx));
end


SigD = Sigma .* Mask;

[VX, DX] = eig(SigD);
DX = diag(DX);
idx = (abs(DX) > max(abs(DX)) * 1e-6);
DX = sqrt(DX(idx));
SigDroot = VX(:,idx) * diag(DX) * (VX(:,idx)');
SigDrootInv = VX(:,idx) * diag(1./DX) * (VX(:,idx)');


% parameter in the augmented lagrangian
rho = 2;
% rho = 1;

B = SigDrootInv * Sigma * SigDrootInv;

A = @(varargin)linop_crosprod( p, p, SigDroot, SigDroot, varargin{:} );

x_cur = 0;
y_cur = zeros(p,p);

[U D] = eig(Sigma);
d = diag(D);
t = cap_soft_th(d, r, tol);
z_cur = U * diag(t) * U';

niter = 0;
initer = 0;

opts = [];
opts.printEvery = Inf;
opts.maxIts = 25;

fprintf('\nCurrent iter: '); 


while 1
    niter = niter + 1;
    initer = initer + 1;
    
    z_old = z_cur;
    Temp = x_cur - y_cur ./ rho +  B ./ rho;
    z_cur = tfocs(smooth_quad, {A, -Temp}, prox_l1(2 * pen * sqrt(log(p)/n) / rho), z_old, opts);
    z_cur = (z_cur + z_cur') / 2;
    
    x_old = x_cur;
    Temp = y_cur ./ rho + A(z_cur, 1);
    [U D V] = svd(Temp, 'econ');
    d = diag(D);
    t = cap_soft_th(d, r, tol);
    x_cur = U * diag(t) * V';
    x_cur = (x_cur + x_cur') / 2;
    
    y_old = y_cur;
    y_cur = y_old + rho .* (A(z_cur,1) - x_cur);
    
    fprintf('%d ', niter); 
    
    if max(rho*norm(x_cur - x_old, 'fro'), norm(z_cur - z_old, 'fro')) < tol
        break
    end
%     if initer == 100;
%         disp(max(norm(x_cur - A(z_cur,1)), rho.*norm(z_cur - z_old)));
%         initer = 0;
%     end
    if niter == maxiter
%         disp(max(norm(x_cur - A(z_cur,1)), rho.*norm(z_cur - z_old)));
        fprintf('\nMaximum number of iterations reached.\n'); 
%         disp('Maximum number of iterations reached.');
        break
    end
end

sol = z_cur;
sol1 = x_cur;









