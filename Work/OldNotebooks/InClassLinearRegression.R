
# The following makes a sequence of values from zero to 20
x<- seq(0,20, .1)

# This makes some noise around a line using our x
Y <- 2+3*x + rnorm(length(x), 0, 1)*5

# Plots the x values against the y values to make a noisy line
plot(x,Y, lwd = .5 , col = "red")

# This calculates the intercept of the regression line.
w0 <- (mean(Y)*mean(x^2) - mean(x)*mean(x*Y))/(mean(x^2) - mean(x)*mean(x))
print(w0)

# This calculates the slope of the regression line.
w1<- (mean(x*Y) - mean(x)*mean(Y))/(mean(x^2) - mean(x)*mean(x))
print(w1)

# This plots the regression line on the same graph we had before.
abline(w0,w1 , lwd = 3 , col = "blue")

# This calculates the R-squared.
R2<- 1- (Y - w0 - w1*x)%*%(Y - w0 - w1*x)/((Y-mean(Y))%*%(Y - mean(Y)))
print(R2)