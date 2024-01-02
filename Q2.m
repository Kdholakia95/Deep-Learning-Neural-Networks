%------------Final code for Ques2-------------------- 
D=importdata("traingroup28.csv");
D=D.data;
X=D(1:315,1:2);
Y2=D(1:315,3) + 1;
Y1=zeros(3,315);
N=315;   %number of data points  

for i=1:N
    Y1(D(i,3)+1,i)=1;
end

n0=2;
n1=4;
n2=4;
n3=3;  

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
range=-70:1:70;
[X5 , Y5]= meshgrid(range,range);
XX1= X5(:);
XX2= Y5(:);
XX=[XX1 XX2];

size = numel(X5);

AA1=zeros(n1,size);
AA2=zeros(n2,size);
AA3=zeros(n3,size);

ZZ1=zeros(n1,size);
ZZ2=zeros(n2,size);
ZZ3=zeros(n3,size);
size=numel(XX1);
maxepochs=10000;

G=[1 2 10 50 800 1500 maxepochs];
g_count = 1;

Beta=0.01;
eta=0.1;
alpha=0.9;
for epoch=1:maxepochs
    error=0;
    for iter=1:N
      trueclass = Y2(iter);
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
      dw1= -eta * dz1 * X(iter,:);
      db1= -eta * dz1;
      
      w3 = w3 + dw3;
      w2 = w2 + dw2;
      w1 = w1 + dw1;
      b3 = b3 + db3;
      b2 = b2 + db2;
      b1 = b1 + db1;
      error = error - log(a3(trueclass));
    end
     error
     if epoch == maxepochs%G(g_count)
         ZZ1 = w1 * XX' + b1;
         AA1 = tanh(Beta * ZZ1);
         ZZ2 = w2 * AA1 + b2;
         AA2 = tanh(Beta * ZZ2);
         ZZ3 = w3 * AA2 + b3;
         for i=1:size
             AA3(:,i) = softmax(ZZ3(:,i));
         end
      %Use breakpoint at line 160 to plot large number of graphs and delete them.      
        Plot2(XX(:,1),XX(:,2),AA1(1,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL1 Node=1', epoch));       
        figure;Plot2(XX(:,1),XX(:,2),AA1(2,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL1 Node=2', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA1(3,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL1 Node=3', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA2(1,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL2 Node=1', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA2(2,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL2 Node=2', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA2(3,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d HL2 Node=3', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA3(1,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d OL Node=1', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA3(2,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d OL Node=2', epoch));
        figure;Plot2(XX(:,1),XX(:,2),AA3(3,:)');xlabel("Feature1");ylabel("Feature2");
        zlabel("Output");title(sprintf('Epoch=%d OL Node=3', epoch)); 
        
        for k = 1:9 
            figure(k);               
            temp=['fig',num2str(epoch),num2str(k),'.png']; 
            saveas(gca,temp); 
        end
            g_count = g_count + 1;
        %} 
    end
end

for a = 1:numel(AA3(1,:))
    [~,id] = max(AA3);
end

Class_1_x = [];
Class_2_x = [];
Class_3_x = [];

for a = 1:numel(AA3(1,:))
    if id(a) == 1
        Class_1_x = [Class_1_x; XX(a,1:2)];
    elseif id(a) == 2
        Class_2_x = [Class_2_x; XX(a,1:2)];
    else
        Class_3_x = [Class_3_x; XX(a,1:2)];
    end
end

C1 = [];
C2 = [];
C3 = [];

for a = 1:450
    if D(a,3) == 1
        C1 = [C1; D(a,1:2)];
    elseif D(a,3) == 2
        C2 = [C2; D(a,1:2)];
    else
        C3 = [C3; D(a,1:2)];
    end
end

hold on
plot(Class_1_x(:,1), Class_1_x(:,2), '.r');
plot(Class_2_x(:,1), Class_2_x(:,2), '.b');
plot(Class_3_x(:,1), Class_3_x(:,2), '.g');
xlabel("Feature 1"); ylabel("Feature 2");
title("Decision regions after training")
hold off

figure;
plot(C1(:,1),C1(:,2),'b.')
hold on
plot(C2(:,1),C2(:,2),'g.')
plot(C3(:,1),C3(:,2),'r.')
xlabel("Feature 1"); ylabel("Feature 2");
title("Distribution of input data")
hold off

function [] = Plot2(A, B, C)         
    
    
    dt = delaunayTriangulation(A,B);
    tri = dt.ConnectivityList;
    Ai = dt.Points(:,1);
    Bi = dt.Points(:,2);
    
    F = scatteredInterpolant(A,B,C);
    
    Ci = F(Ai,Bi);
    
    trisurf(tri,Ai,Bi,Ci)    
    shading interp
end