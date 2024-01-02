D = load("train100 - train100.csv");
X = D(:,1:2);
Y2 =  D(:,3) ;
%Y1=zeros(3,451);
N = numel(Y2);   %number of data points  



n0 = 2;
n1 = 3;
n2 = 3;
n3 = 1;

w1 = rand(n1,n0);
w2 = rand(n2,n1);
w3 = rand(n3,n2);

b1 = rand(n1,1);
b2 = rand(n2,1);
b3 = rand(n3,1);

z1 = zeros(n1,1);
z2 = zeros(n2,1);
z3 = zeros(n3,1);

a1 = zeros(n1,1);
a2 = zeros(n2,1);
a3 = zeros(n3,1);

da1 = zeros(n1,1);
da2 = zeros(n2,1);
da3 = zeros(n3,1);

dz1 = zeros(n1,1);
dz2 = zeros(n2,1);
dz3 = zeros(n3,1);

dw1=zeros(n1,n0);
dw2=zeros(n2,n1);
dw3=zeros(n3,n2);

db1=zeros(n1,1);
db2=zeros(n2,1);
db3=zeros(n3,1);

dh3=zeros(n3,1);

dw1prev=zeros(n1,n0);
dw2prev=zeros(n2,n1);
dw3prev=zeros(n3,n2);

db1prev=zeros(n1,1);
db2prev=zeros(n2,1);
db3prev=zeros(n3,1);

range=-100:1:50;
[X5 , Y5]= meshgrid(range,range);
XX1 = X5(:);
XX2= Y5(:);
XX=[XX1 XX2];

size = numel(X5);

maxepochs=10000;
Beta=0.01;
eta=0.01;
alpha=0.9;
Error_each_epoch = zeros(maxepochs,2);
for epoch=1:maxepochs
    error=0;
    for iter=1:N
      %trueclass = Y2(iter);
      dw1prev=dw1;
      dw2prev=dw2;
      dw3prev=dw3;
      db1prev=db1;
      db2prev=db2;
      db3prev=db3;
      
      z1 = w1 * X(iter,:)' + b1;
      a1 = tanh(Beta*z1);
      
      z2 = w2*a1 +b2;
      a2 = tanh(Beta*z2);
      
      z3 = w3*a2 + b3;
      a3 = z3;
      

      dz3 = a3-Y2(iter);
      dw3 = -eta * dz3 * (a2');
      db3 = -eta * dz3;
      
      da2= w3' * dz3;
      dz2= Beta .* da2 .* (1-(a2.^2));
      dw2= -eta * dz2 * a1';
      db2= -eta * dz2;
      
      da1= w2' * dz2; 
      dz1= Beta .* da1 .* (1-(a1.^2));
      dw1= -eta * dz1 * X(iter,:);
      db1= -eta * dz1;
      
      w3 = w3 + dw3 + alpha*dw3prev;
      w2 = w2 + dw2 + alpha*dw2prev;
      w1 = w1 + dw1 + alpha*dw1prev;
      b3 = b3 + db3 + alpha*db3prev;
      b2 = b2 + db2 + alpha*db2prev;
      b1 = b1 + db1 + alpha*db1prev;
       error = error + (a3-Y2(iter))^2;
    end
    error = error/100;
    Error_each_epoch(epoch,1) = epoch;
    Error_each_epoch(epoch,2) = error;
    error
    if(epoch==10000)
        ZZ1 = w1 * XX' + b1;
         AA1 = tanh(Beta * ZZ1);
         ZZ2 = w2 * AA1 + b2;
         AA2 = tanh(Beta * ZZ2);
         ZZ3 = w3 * AA2 + b3;
         AA3 = ZZ3;
         figure;Plot2(XX(:,1),XX(:,2),AA3(1,:)');
         xlabel("Feature1");
         ylabel("Feature2");
         zlabel("Approximated Value");
         title("Plot of Approximated Function");
    end
end
figure;plot(Error_each_epoch(:,1),Error_each_epoch(:,2),'g-o');
xlabel("Epoch Number");
ylabel("Error");
title("Plot of Error Vs Epoch");