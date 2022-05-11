# Function for Intialization via generalized fantope projection
# Inputs:
# =======
# A, B:       Pair of matrix to calculate generalized eigenspace (sigma, sigma0)
# nu:         Parameter of ADMM, default set to 1
# K:          nuclear norm constraint, equal to r
# rho:     penalty parameter on the l_1 norm of the solution, scaled by
#             sqrt(log(max(p1,p2))/n)
# epsilon:    tolerance level for convergence in ADMM
# maxiter:    maximum number of iterations in ADMM
# trace:      if set to True will print all iterations 

# Outputs:
# ========
# $Pi:     optimum of the convex program

sgca_init <-
  function(A,B,rho,K,nu=1,epsilon=5e-3,maxiter=1000,trace=FALSE){
    p <- nrow(B)
    eigenB <- eigen(B)
    sqB <- eigenB$vectors%*%sqrt(diag(pmax(eigenB$values,0)))%*%t(eigenB$vectors)	
    tau <- 4*nu*eigenB$values[1]^2	
    criteria <- 1e10
    i <- 1
    # Initialize parameters
    H <- Pi <- oldPi <-  diag(1,p,p)
    Gamma <- matrix(0,p,p)
    # While loop for the iterations
    while(criteria > epsilon && i <= maxiter){
      for (i in 1:20){
        Pi <- updatePi(B,sqB,A,H,Gamma,nu,rho,Pi,tau)
      }
      #Pi <- updatePi(B,sqB,A,H,Gamma,nu,lambda,Pi,tau)
      
      H <- updateH(sqB,Gamma,nu,Pi,K)
      Gamma <- Gamma + (sqB%*%Pi%*%sqB-H) * nu	
      criteria <- sqrt(sum((Pi-oldPi)^2))
      oldPi <- Pi
      i <- i+1
      if(trace==TRUE)
      {
        print(i)
        print(criteria)
      }
    }
    return(list(Pi=Pi,H=H,Gamma=Gamma,iteration=i,convergence=criteria))
    
  }
