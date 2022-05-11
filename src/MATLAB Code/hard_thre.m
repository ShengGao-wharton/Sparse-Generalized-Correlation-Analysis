function U_hard = hard_thre( U, s )

% U is the matrix. s is number of nonzero rows we want.
% Do hard thresholding on rows of U.

if size(U,1) <= s
    U_hard = U;
else

    U_row = sqrt(sum(abs(U).^2,2));
    [~,max_id] = sort(U_row,'descend');
    max_want = max_id(1:s);

    U_hard = zeros(size(U));
    U_hard(max_want,:) = U(max_want,:);
end

end

