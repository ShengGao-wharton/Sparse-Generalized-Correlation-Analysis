gca_to_cca <-
  function(a_estimate, S, pp){
    p1 = pp[1];
    p2 = pp[2];
    p = p1 + p2;
    sigmaxhat = S[1:p1,1:p1];
    sigmayhat = S[(p1+1):p,(p1+1):p];
    u_estimate = a_estimate[1:p1,] %*% sqrtm(t(a_estimate[1:p1,]) %*% sigmaxhat %*% a_estimate[1:p1,])$Binv;
    v_estimate = a_estimate[(p1+1):p,] %*% sqrtm(t(a_estimate[(p1+1):p,]) %*% sigmayhat %*% a_estimate[(p1+1):p,])$Binv;
    l = list("u" = u_estimate, "v" = v_estimate)
    return(l)
  }