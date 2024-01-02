Xtest = zeros(350,newdimension);
Ytest = zeros(350,1);
Xtest(1:70,:)=XtrainNew(231:300,:);
Xtest(71:140,:)=XtrainNew(531:600,:);
Xtest(141:210,:)=XtrainNew(831:900,:);
Xtest(211:280,:)=XtrainNew(1131:1200,:);
Xtest(281:350,:)=XtrainNew(1431:1500,:);
Ytest(1:70,:) = 1;
Ytest(71:140,:) = 2;
Ytest(141:210,:) = 3;
Ytest(211:280,:) = 4;
Ytest(281:350,:) = 5;
Ntest=350;
A11=zeros(n1,Ntest);
H11=zeros(n1,Ntest);
A22=zeros(n2,Ntest);
H22=zeros(n2,Ntest);
A33=zeros(n3,Ntest);
H33=zeros(n3,Ntest);

A11= w1 * Xtest' + b1;
for i=1:Ntest
    H11(:,i)= tanh(Beta*A11(:,i));
end

A22= w2*H11 + b2;
for i=1:Ntest
    H22(:,i)= tanh(Beta*A22(:,i));
end

A33= w3 * H22 + b3;

for i=1:Ntest
    H33(:,i)=softmax(A33(:,i));
end
Confusion_matrix = zeros(n3,n3);
yhat=zeros(350,1);
max=-1;
maxidx=1;
Yhat = zeros(1,Ntest);
count=0;
for j=1:Ntest
    max= H33(1,j);
    maxidx=1;
    for i=2:5
        if H33(i,j)>max
            max=H33(i,j);
            maxidx=i;
        end
    end
    Yhat(j)=maxidx;
    if Yhat(j)==Ytest(j)
        count=count+1;
    end
    Confusion_matrix(Ytest(j),Yhat(j)) = Confusion_matrix(Ytest(j),Yhat(j)) + 1;
end