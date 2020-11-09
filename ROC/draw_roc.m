

%draw roc curve with probility mat


clear;


type_all={'no_atten'};%,'rebuttal_finetune_attention_rimone_5000'};fc2_20000
%root = '..\txt_result\12.20\';%final\rename\';
root = './after_process/';
%type_all={'alex_115','alex_832','attention_115','attention_832','v3_115','v3_832'};
% type_all={'fc2_25500'};%{'attention_model5_without_attention_W60_decay3_10000','attention_model5_without_attention_W60_decay3_45000'};
% %type_all={'attention_model5_without_multi_scale_W60_decay3','attention_model5_without_attention_W60_decay3_45000' };
% root = '..\txt_result\rimone\';%final\rename\';
result=zeros(3,length(type_all));

for type_num =1%:2%:length(type_all)
    
    type=type_all{1,type_num};

load([root,type,'_0.mat']);
load([root,type,'_1.mat']);
TT=0.5;
posi_num=length(data_positive);%Ϊ�������
neg_num=length(data_negative);%Ϊ�����

a=zeros(2,length(data_positive)+length(data_negative));


sstep=0.001;
threshold=0:sstep:1;%��101������,���ڸ����ޣ��ж�Ϊ��
data=zeros(length(threshold),3);  %format=[threshold,precision,recall]

%������Ϊ ����� positive glaucoma  &�������negative glaucoma
%ҽѧ��Ϊpositive ����Ϊ����
for k= 1:length(threshold)
    
    data(k,1)=threshold(k);
    TP=0;%true positive ��������Ϊ��עΪ����ۣ��ж���ȷ���ж�Ϊ�����
    FN=0;%false negtive �������ж�Ϊ�������д���עΪ�����
    FP=0;%false positive �������ж�Ϊ����ۣ��жϴ��󣬱�עΪ����
    TN=0;%true negtive ��������עΪ������ۣ�ʵ��Ϊ����
    	for num=1:posi_num  %ʵ��0������
            if data_positive(num,4) >= threshold(k)  %��������ֵ���ж�Ϊ�����
                FP=FP+1;
            else
                TN=TN+1;%������Ϊ��
            end
        end
        
        
       for j=1:neg_num
            if data_negative(j,4) >= threshold(k)  %������Ϊ��
                 TP=TP+1;
            else
                 FN=FN+1;%������Ϊ��
            end
       end
        
       TPR=TP/(TP+FN);%true positive rate,����y,RECALL
       FPR=FP/(FP+TN);%false positive rate������x  %������Ϊ�в��ģ�ռ����û���ı���
       
       if threshold(k)==TT
         recall=TPR;%sensitivity
         precision=TP/(TP+FP);%����в���ռ���б���Ϊ�в��ı�����������balance����
      	 specificity=1-FPR;
         TP
         TN
         FP
         FN
         acc = (TP+TN)/(TP+TN+FP+FN)
       end
       
       data(k,2)=TPR;%y  %y,sensi
       data(k,3)=FPR;%x
end

% save('../construct/roc_param_01-14-60000.mat','data');

%��roc����
x=100*data(:,3);
y=100*data(:,2);

auc=0;
for i=(1+(1/sstep)):-1:2  %y��������
    s_h=(y(i)+y(i-1))*0.5;
    s_w=x(i-1)-x(i);
    auc=auc+s_h*s_w;
end
auc = auc / 10000;

figure(10)
%figure('visible','off')
plot(x,y,'-r','Linewidth',2)
xlabel('1 - Specificity','fontsize',12);% x�����?
ylabel('Sensitivity','fontsize',12); 
axis([0,100,0,100]);

set(gca,'XTick',[0:0.2:1]);%����Ҫ��ʾ����̶�

set(gca,'YTick',[0:0.2:1]);%����Ҫ��ʾ����̶�


title('ROC Curve ','fontsize',12);

hold on

plot([0,1],[0,1],'--b')

wrong_neg=0;
for num=1:length(data_negative)
   if data_negative(num,3)>= data_negative(num,4)
       wrong_neg=wrong_neg+1;
       num;
   end
end
wrong_neg

wrong_posi=0;
for num=1:length(data_positive)
   if data_positive(num,3)< data_positive(num,4)
       wrong_posi=wrong_posi+1;
       num;
   end
end
wrong_posi


%k SCORE
F_score=2*precision*recall/(precision+recall)
F2_score=5*precision*recall/(4*precision+recall)
auc
sensitivity=recall
%acc = 1-(wrong_neg+wrong_posi)/
specificity
hold on
plot(1-specificity,sensitivity,'dr','markersize',5,'Linewidth',5)


result(1,type_num)=auc;
result(2,type_num)=sensitivity;
result(3,type_num)=specificity;
hold on
end
% 
x_ori=fliplr(round(x'*1000)/10);
y_ori=fliplr(round(y'*1000)/10);
fix = [round((1-specificity)*1000)/10;round(sensitivity*1000)/10];
% 
% 
% 
% %�����������ݣ�����data_final�����Ƶ�excel���������ͼ
% num=1/sstep +1;
% data_low =zeros(1,num);
% data_high =zeros(1,num);
% 
% beta = 1/sstep/100;
% 
% for n_after=1:num    %data(num)
%     for n_ori = 1:num
%         if x_ori(n_ori)==(n_after-1)/beta         %��������ԭ���������У�ֱ�Ӹ�ֵ������Ҫ��ֵ
%             data_low(n_after)=y_ori(n_ori);
%             break
%         end    
%     end
%     
%     for n_ori =1 : num-1
%         for n_ori_high = n_ori : num-1
%             if x_ori(n_ori)==(n_after-1)/beta  &&  x_ori(n_ori_high)==(n_after-1)/beta  &&   x_ori(n_ori_high+1)~=(n_after-1)/beta  %��������ԭ���������У�ֱ�Ӹ�ֵ������Ҫ��ֵ
%                 data_high(n_after)=y_ori(n_ori_high);
%                 break
%             end 
%         end
%     end
%     
% end
% data_high(num)=data_low(num);
% 
% %��data����0Ԫ�ؽ��в�ֵ
% 
% flag_num = sum(data_low~=0)+1;
% flag_fix = zeros(1,flag_num);
% flag_fix(1)=1;
% i=1;
% data_final=data_low;
% for a = 1:num
%     if data_low(a)~=0
%         i=i+1;
%         flag_fix(i)= a;
%         
%     end
% end
%         
% 
% for region_NO = 1:flag_num-1
%     region_low = flag_fix(region_NO);
%     region_high = flag_fix(region_NO+1);
%     for n= region_low+1:region_high-1
%         data_final(n)= (data_low(region_high)-data_high(region_low))*(n-region_low)/(region_high-region_low)+data_high(region_low);
%         
%     end
% end
% 
% acu1 = sum(data_final)*0.001*0.01





    
