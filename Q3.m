%--------------Final code for Ques3-------------
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
% npoints_each_class = 300;
% dimension = 512;
% Cov = zeros(dimension,dimension);
% mean = zeros(1,dimension);
% 
% for i=1:npoints
%     mean = mean + Xtrain(i,:);
% end  
% mean = mean/npoints;
% 
% for i=1:npoints
%     Cov = Cov + (Xtrain(i,:)-mean(1,:))' * (Xtrain(i,:)-mean(1,:));
% end
% Cov=Cov/npoints;
% 
% newdimension = 20;
% [V,D] = eig(Cov);
% [B,id] = sort(diag(D),'descend');
% V = V(:,id);
% EigVec = zeros(dimension,newdimension);
% EigVec(:,1:newdimension) = V(:,1:newdimension); %Extracting top 20 eig vectors
% 
% XtrainNew = Xtrain * EigVec;   % projecting the data on to eigen vectors
% YtrainNew = Ytrain;            % y reamins same as previous one
x1 = zeros(300,20);
x2 = zeros(300,20);
x3 = zeros(300,20);
x4 = zeros(300,20);
x5 = zeros(300,20);
dimension = 512;
Cov = zeros(dimension,dimension);
mean = zeros(1,dimension);
npoints = 300;      %number of points in each class;
index=0;
newdimension=20;
X = zeros(1500,20);
for j=1:5
    Cov=zeros(512,512);
    index=(j-1)*300;
    x2 = Xtrain(index+1:index+300,:);
    for k=1:npoints
        mean = mean + x2(k,:);
    end
    mean = mean/npoints;
   for i=1:npoints
      Cov = Cov + (x2(i,:)-mean(1,:))' * (x2(i,:)-mean(1,:));
   end
   Cov=Cov/npoints;
   
   [V,D] = eig(Cov);
   [B,id] = sort(diag(D),'descend');
   V = V(:,id); 
   EigVec = zeros(dimension,newdimension);
   EigVec(:,1:newdimension) = V(:,1:newdimension);
   x1=Xtrain(index+1:index+300,:) * EigVec;
   X(index+1:index+300,:)=x1;
end

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

w1=rand(n1,n0);
w2=rand(n2,n1);
w3=rand(n3,n2);

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

maxepochs=1500;
Beta=0.01;
eta=0.01;
alpha=0.01;
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
      
      w3 = w3 - eta*dw3 - alpha*dw3prev;
      w2 = w2 - eta*dw2 - alpha*dw2prev;
      w1 = w1 - eta*dw1 - alpha*dw1prev;
      b3 = b3 - eta*db3 - alpha*db3prev;
      b2 = b2 - eta*db2 - alpha*db2prev;
      b1 = b1 - eta*db1 - alpha*db1prev;
       error = error - log(a3(trueclass));
    end
    error
 end