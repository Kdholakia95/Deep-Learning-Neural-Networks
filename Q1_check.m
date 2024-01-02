testd=load("val - val.csv");
Xtest=zeros(300,2);
Xtest=testd(:,1:2);
Ytest=zeros(300,1);

Ytest=testd(:,3);
A11=zeros(n1,300);
H11=zeros(n1,300);
A22=zeros(n2,300);
H22=zeros(n2,300);
A33=zeros(1*300);
A11=w1 * Xtest' + b1;
for i=1:300
    H11(:,i) = tanh(Beta*A11(:,i));
end

A22= w2 * H11 + b2;

for i=1:300
    H22(:,i)=tanh(Beta*A22(:,i));
end

A33= w3 * H22 + b3;

A33=A33';
testerr=0;
for i=1:300
    testerr=testerr+(Ytest(i)-A33(i))^2;
end

testerr=testerr/300;
testerr
XX=-120:0.000001:15;

plot(XX,XX,'g.');
hold on
scatter(Ytest,A33,'bo');
xlim([-120 15])
ylim([-120 15])
xlabel("Actual value of Y");
ylabel("Predicted value of Y");
title("Scatter plot of Actual Y vs Predicted Y");