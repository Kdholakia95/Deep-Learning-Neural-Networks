%---------------initialisation------------------------
Y1=zeros(5,1150);
N=1150;   %number of data points  

for i=1:N
    Y1(Ynew(i,1),i)=1;
end

n0=newdimension;
n1=5;
n2=5;
n3=5;  

w1=w11;
w2=w22;
w3=w33;
b1=b11;
b2=b22;
b3=b33;

z1=zeros(n1,1);
z2=zeros(n2,1);
z3=zeros(n3,1);

a1=zeros(n1,1);
a2=zeros(n2,1);
a3=zeros(n3,1);

da1=zeros(n1,1);
da2=zeros(n2,1);
da3=zeros(n3,1);

dz1=zeros(n1,1);
dz2=zeros(n2,1);
dz3=zeros(n3,1);

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

confusion_matrix_train=zeros(5,5);
maxepochs=5000;
Error_each_epoch = zeros(maxepochs,2);
Beta=.01;
eta=0.01;
alpha=0.9;
for epoch=1:maxepochs
    error=0;
    for iter=1:N
      trueclass = Ynew(iter);
      dw1prev=dw1;
      dw2prev=dw2;
      dw3prev=dw3;
      db1prev=db1;
      db2prev=db2;
      db3prev=db3;
      
      z1 = w1 * Xnew(iter,:)' + b1;
      a1 = tanh(Beta*z1);
      
      z2 = w2 * a1 + b2;
      a2 = tanh(Beta * z2);
      
      z3 = w3 * a2 + b3;
      a3 = softmax(z3);
      
      dz3 = a3-Y1(:,iter);
      dw3 = -eta * dz3 * (a2');
      db3= -eta * dz3;
      
      da2= w3' * dz3;
      dz2= Beta .* da2 .* (1-(a2.^2));
      dw2= -eta * dz2 * a1';
      db2= -eta * dz2;
      
      da1= w2' * dz2;
      dz1= Beta .* da1 .* (1-(a1.^2));
      dw1= -eta * dz1 * Xnew(iter,:);
      db1= -eta * dz1;
      
      w3 = w3 + dw3 + alpha * dw3prev;
      w2 = w2 + dw2 + alpha * dw2prev;
      w1 = w1 + dw1 + alpha * dw1prev;
      b3 = b3 + db3 + alpha * db3prev;
      b2 = b2 + db2 + alpha * db2prev;
      b1 = b1 + db1 + alpha * db1prev;
      error = error - log(a3(trueclass));
      if(epoch==maxepochs)
          maxprob=-1;
          label=7;
          for a=1:5
              if(a3(a)>maxprob)
                  maxprob=a3(a);
                  label=a;
              end
          end
          confusion_matrix_train(trueclass,label) = confusion_matrix_train(trueclass,label)+1;
      end
    end
    %error = error/100;
    Error_each_epoch(epoch,1) = epoch;
    Error_each_epoch(epoch,2) = error;
    error
   
end
figure;plot(Error_each_epoch(:,1),Error_each_epoch(:,2));
xlabel("Epoch Number");
ylabel("Error");
title("Plot of Error Vs Epoch for Generalised delta rule");