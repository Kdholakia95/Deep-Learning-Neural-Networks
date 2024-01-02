%--------------Final code for Ques3-------------
%{
imcar = readNPY('car.npy');
imcat = readNPY('cat.npy');
imdog = readNPY('dog.npy');
imship = readNPY('ship.npy');
imtruck = readNPY('truck.npy');

Xtrain=zeros(1500,512);
Xtrain(1:300,:) = imcar;
Xtrain(301:600,:) = imcat;
Xtrain(601:900,:) = imdog;
Xtrain(901:1200,:) = imship;
Xtrain(1201:1500,:) = imtruck;

Ytrain=zeros(1500,1);
Ytrain(1:300,1) = 1;
Ytrain(301:600,1) = 2;
Ytrain(601:900,1) = 3;
Ytrain(901:1200,1) = 4;
Ytrain(1201:1500,1) = 5;

%-------------------------apply PCA----------------------------

dimension = 512;

Cov = cov(Xtrain);
%Cov=Cov/1500;

newdimension = 20;
[V,D] = eig(Cov);
[B,id] = sort(diag(D),'descend');
V = V(:,id);
EigVec = zeros(dimension,newdimension);
EigVec(:,1:newdimension) = V(:,1:newdimension); %Extracting top 20 eig vectors

X = Xtrain * EigVec;   % projecting the data on to eigen vectors
YtrainNew = Ytrain;            % y reamins same as previous one
% x1 = zeros(300,20);
% x2 = zeros(300,20);
% x3 = zeros(300,20);
% x4 = zeros(300,20);
% x5 = zeros(300,20);
% dimension = 512;
% Cov = zeros(dimension,dimension);
% mean = zeros(1,dimension);
% npoints = 300;      %number of points in each class;
% index=0;
% newdimension=20;
% X = zeros(1500,20);
% for j=1:5
%     Cov=zeros(512,512);
%     index=(j-1)*300;
%     x2 = Xtrain(index+1:index+300,:);
%     for k=1:npoints
%         mean = mean + x2(k,:);
%     end
%     mean = mean/npoints;
%    for i=1:npoints
%       Cov = Cov + (x2(i,:)-mean(1,:))' * (x2(i,:)-mean(1,:));
%    end
%    Cov=Cov/npoints;
%    
%    [V,D] = eig(Cov);
%    [B,id] = sort(diag(D),'descend');
%    V = V(:,id); 
%    EigVec = zeros(dimension,newdimension);
%    EigVec(:,1:newdimension) = V(:,1:newdimension);
%    x1=Xtrain(index+1:index+300,:) * EigVec;
%    X(index+1:index+300,:)=x1;
% end

%X = pca(Xtrain);

Xnew=zeros(1150,newdimension);
Xnew(1:230,:) = X(1:230,:);
Xnew(231:460,:) = X(301:530,:);
Xnew(461:690,:) = X(601:830,:);
Xnew(691:920,:) = X(901:1130,:);
Xnew(921:1150,:) = X(1201:1430,:);

Ynew=zeros(1150,1);
Ynew(1:230,1) = 1;
Ynew(231:460,1) = 2;
Ynew(461:690,1) = 3;
Ynew(691:920,1) = 4;
Ynew(921:1150,1) = 5;

%}
%--------------------------------------------------------------------------

%-------------------------Nueral Network-----------------------------------
    % Here newdimension is the reduced dimension of the data 
    % Number of classes are five
    
Y1=zeros(5,1150);
N=1150;   %number of data points  

for i=1:N
    Y1(Ynew(i,1),i)=1;
end



n0=newdimension;
n1=5;
n2=5;
n3=5;    % Total five classes in the dataset

w1=w11;
w2=w22;
w3=w33;

b1=rand(n1,1);
b2=rand(n2,1);
b3=rand(n3,1);

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

q1=zeros(n1,n0);
q2=zeros(n2,n1);
q3=zeros(n3,n2);

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

maxepochs=500;
Beta=0.01;
eta=0.01;
alpha=0.1;
epsl = 0.001;
p1 = 0.9;
p2 = 0.999;
r1=0;r2=0;r3=0;
q1=0;q2=0;q3=0;
r1b=0;r2b=0;r3b=0;
q1b=0;q2b=0;q3b=0;

pl =zeros(maxepochs,2)
 m = 1;
for epoch=1:maxepochs
    error=0;
   
    pl(epoch,1) = epoch;
    
    %_______r for w and b
    rw1_sum=0;
    rw2_sum=0;
    rw3_sum=0;
    rb1_sum=0;
    rb2_sum=0;
    rb3_sum=0;
    
    
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
      
      z2 = w2*a1 +b2;
      a2 = tanh(Beta*z2);
      
      z3 = w3*a2 + b3;
      a3 = softmax(z3);
      
      dz3 = a3-Y1(:,iter);
      dw3 = dz3 * (a2');
      db3=dz3;
      
      da2= w3' * dz3;
      dz2= Beta .* da2 .* (1-(a2.^2));
      dw2= dz2 * a1';
      db2= dz2;
      
      da1= w2' * dz2;
      dz1= Beta .* da1 .* (1-(a1.^2));
      dw1= dz1 * Xnew(iter,:);
      db1= dz1;
      
      
      q1 = (p1 .* q1) + (1-p1).* dw1;
      q2 = (p1 .* q2) + (1-p1).* dw2;
      q3 = (p1 .* q3) + (1-p1).* dw3;
      
      
      r1 = p2 .* r1 + (1-p2) .* (dw1.^2);
      r2 = p2 .* r2 + (1-p2) .* (dw2.^2);
      r3 = p2 .* r3 + (1-p2) .* (dw3.^2);
      
      qcap1 = q1 ./ (1-(p1)^m);
      qcap2 = q2 ./ (1-(p1)^m);
      qcap3 = q3 ./ (1-(p1)^m);
      
      rcap1 = r1 ./ (1-(p2)^m);
      rcap2 = r2 ./ (1-(p2)^m);
      rcap3 = r3 ./ (1-(p2)^m);
      
      
%       for bias
      q1b = (p1 .* q1b) + (1-p1).* db1;
      q2b = (p1 .* q2b) + (1-p1).* db2;
      q3b = (p1 .* q3b) + (1-p1).* db3;
      
      
      r1b = p2 .* r1b + (1-p2) .* (db1.^2);
      r2b = p2 .* r2b + (1-p2) .* (db2.^2);
      r3b = p2 .* r3b + (1-p2) .* (db3.^2);
      
      qcap1b = q1b ./ (1-(p1)^m);
      qcap2b = q2b ./ (1-(p1)^m);
      qcap3b = q3b ./ (1-(p1)^m);
      
      rcap1b = r1b ./ (1-p2^m);
      rcap2b = r2b ./ (1-p2^m);
      rcap3b = r3b ./ (1-p2^m);
      m = m + 1;
      
      %__________delta w and b
      for a = 1:5
          for b = 1:20
              delw1(a,b) = (eta * qcap1(a,b))./(epsl + (rcap1(a,b)^0.5));
          end
      end
      
      for a = 1:5
          for b = 1:5
              delw2(a,b) = (eta * qcap2(a,b))./(epsl + (rcap2(a,b)^0.5));
              delw3(a,b) = (eta * qcap3(a,b))./(epsl + (rcap3(a,b)^0.5));
          end    
      end
      for a = 1:5
        delb1 = (eta * qcap1b(a,1))./(epsl + (rcap1b(a))^0.5);
        delb2 = (eta * qcap2b(a,1))./(epsl + (rcap2b(a))^0.5);
        delb3 = (eta * qcap3b(a,1))./(epsl + (rcap3b(a))^0.5);
      end
      
      
      w3 = w3 - delw3;
      w2 = w2 - delw2;
      w1 = w1 - delw1;
      b3 = b3 - delb1;
      b2 = b2 - delb2;
      b1 = b1 - delb3;
      error = error - log(a3(trueclass));
  
       
%        for ada grade
%        rw1_sum = rw1_sum + (dw1).^2;
%        rw2_sum = rw2_sum + (dw2).^2;
%        rw3_sum = rw3_sum + (dw3).^2;
%        
%        rb1_sum = rb1_sum + (db1).^2;
%        rb2_sum = rb2_sum + (db2).^2;
%        rb3_sum = rb3_sum + (db3).^2;
       
    end
    error
    pl(epoch,2) = error;
end

plot(pl(:,1),pl(:,2));
xlabel("Epoch Number");
ylabel("Error");
title("Plot of Error Vs Epoch for Adaptive moments method rule");