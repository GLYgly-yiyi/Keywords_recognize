clear all;
%melfilterbank Compute the mel filterbank. 
            % Compute the minimum and maximum mels
            %÷���˲�����ʼ��
            sampling_frequency=16000; %����Ƶ��
            window_length=512; %������
            number_filters=64; %�����÷���˲�����
            %mininum_melfrequency = 2595*log10(1+(sampling_frequency/window_length)/700);
            mininum_melfrequency = 0;%��С÷��Ƶ��
            maximum_melfrequency = 2595*log10(1+(sampling_frequency/2)/700); %���÷��Ƶ��
            
%             xxxx = linspace(mininum_melfrequency,16000,16000);%�ȼ���ķֳ�66���� ��ΪҪ�����߽� ���Զ���2��
%             yyyy = 2595*log10(1+(xxxx/2)/700); %���÷��Ƶ��
%             figure(1)
%             plot(yyyy ,'LineWidth',2.5);
%             xlabel('����Ƶ��/Hz','fontsize',15);
%             ylabel('÷��Ƶ��/Hz','fontsize',15); 
            % Derive the width of the half-overlapping filters in the mel scale (constant)
            %filter_width = 2*(maximum_melfrequency-mininum_melfrequency)/(number_filters+1);
            filter_indices = linspace(mininum_melfrequency,maximum_melfrequency,66);%�ȼ���ķֳ�66���� ��ΪҪ�����߽� ���Զ���2��
            filter_indices = filter_indices';%ת�� 1,66 to  66,1
            
            % Compute the start and end indices of the overlapping filters in the mel scale (linearly spaced)
            %filter_indices = mininum_melfrequency:filter_width/2:maximum_melfrequency;
            
            % Derive the indices of the filters in the linear frequency scale (log spaced)
            %��66��÷��Ƶ��ת��ΪƵ�ʺ��� �ܵ�512���ʾ1.6KHZ �е�λ������ ����[41;44;46;49;52;55;58;61;64;68]
            NFFT =512;
            filter_indices = floor(700*(10.^(filter_indices/2595)-1)*(NFFT+1)/sampling_frequency);
            
            % Initialize the mel filterbank
            mel_filterbank = zeros(number_filters,window_length/2+1);
            mel_filterbank_in = zeros(number_filters,window_length/2);
                        
            % Loop over the filters ���ݹ�ʽ����÷���˲���
            for j = 1:number_filters
                for i = filter_indices(j):filter_indices(j+1)
                        mel_filterbank(j,i+1) = (i - filter_indices(j)) / (filter_indices(j+1)-filter_indices(j));
                end
                for i = filter_indices(j+1):filter_indices(j+2)
                        mel_filterbank(j,i+1) = (filter_indices(j+2)-i) / (filter_indices(j+2)-filter_indices(j+1));
                end
            end
            % Make the mel filterbank sparse
            %mel_filterbank = sparse(mel_filterbank);
            mel_filterbank_in(:,:) = mel_filterbank(:,1:256);
            mel_filterbank_in = mel_filterbank_in';
   figure(1)
   plot(mel_filterbank,'LineWidth',1);
   title('Mel÷���˲�����')
   
%�����������          
% x=linspace(0,window_length-1,window_length);
% w = 0.54 - 0.46 * cos(2 * 3.1415926535 * (x) / (window_length - 1) ) ;% ������
%  figure(2)
%  plot(w,'b');
%  title('������')
%  one =randsrc(1,1,randperm(9))
%  one=num2str(one);
%  two =randsrc(1,1,randperm(9))
%  two=num2str(two);
%  three_panduan=randsrc(1,1,randperm(10));
%  if (three_panduan>5)
%      three='A';
%  else
%      three='I';
%  end
%  str=['G:\��Ƶ���ݼ�\ST-CMDS-20170001_1-OS\20170001P0000' one three '000' two '.wav'];
 %[input,Fs]=audioread(str);
 
%[input,Fs]=audioread('G:\gly\matlab\matlab_800_64\��������\10.wav');
%[input,Fs]=audioread('G:\��Ƶ���ݼ�\ST-CMDS-20170001_1-OS\20170001P00241I0001.wav');
[input,Fs]=audioread('G:\peprocess\test\left.wav'); 
% [input,Fs]=audioread('G:\gly\keywords\train\audio\down\0b77ee66_nohash_0.wav'); 
% input = input';
% jieduan_input = zeros(16000,1);
% jieduan_input = input(12001:28000,1);
% input  =jieduan_input;
 input = (input*32768)';

 

% fid1=fopen('G:\peprocess\vad\new_vad\new_vad\coe\input.coe','w+');
% fprintf(fid1,'memory_initialization_radix=10;');
% fprintf(fid1,'memory_initialization_vector=');
% for t=1:size(input,2)
%     fprintf(fid1,'%d',input(1,t));
%     fprintf(fid1,',\n');
% end
% for t=1:4000
%     fprintf(fid1,'%d',input(1,t));
%     fprintf(fid1,',\n');
% end
% [input,Fs]=audioread('G:\peprocess\test\z_my_left1.wav'); 
%  input = (input*32768)';
%  for t=1:4000
%     fprintf(fid1,'%d',input(1,t));
%     fprintf(fid1,',\n');
% end
% for t=1:size(input,2)
%     fprintf(fid1,'%d',input(1,t));
%     fprintf(fid1,',\n');
% end
% for t=1:4000
%     fprintf(fid1,'%d',input(1,t));
%     fprintf(fid1,',\n');
% end
% fclose(fid1);
 
 
[shengdaoshu ,caiyangdianshu] = size(input);% Nnn:��������
%Fs = 16000;
padlen_emphasized_input = (100-1)*160+512;%�����ݽ��в��������㣩128352
padlen_input_emphasized_input = zeros(1,padlen_emphasized_input);
padlen_input = zeros(1,padlen_emphasized_input);
padlen_input(1,1:caiyangdianshu)=input(1,:);

erjinzhi =  zeros(16,padlen_emphasized_input);
%Ԥ����
min(padlen_input,[],'all')
emphasized_input = zeros(1,caiyangdianshu);
for i = 1:padlen_emphasized_input-1
    padlen_input_emphasized_input(i+1) = padlen_input(i+1)-0.96875*padlen_input(i);
end
padlen_input_emphasized_input(1) = padlen_input(1);

figure(3)
subplot(211);
plot(padlen_input);
 title('ԭʼ�ź�')
 subplot(212);
plot(padlen_input_emphasized_input);
 title('Ԥ���غ�')

time_window = 32 ;%�� λms
range0_end = ((padlen_emphasized_input)/Fs*1000 - time_window) / 10 +1  ; %����ѭ����ֹ��λ�ã�Ҳ�����������ɵĴ���
range0_end = round(range0_end)
data_input = zeros(100, window_length/2,1);% ���ڴ�����յ�Ƶ����������
data_output = zeros(100, 64,1);% ���ڴ�����յ�Ƶ����������
sgn=zeros(1,100);
sgn_flag_1=0;
sgn_flag_2=0;
sgn_jishu=0;
sgn_jishu_2=0;
sgn_x1=0;
sgn_x2=0;
sgn_ths=0;

senergy=zeros(1,100);
senergy_flag_1=0;
x1=0;
x2=0;
senergy_jishu_1=0;
senergy_jishu_2=0;
senergy_flag_2=0;
senergy_ths=0;

scorr=zeros(1,100);

start=0;

x=linspace(0,window_length-1,window_length);
w = 0.54 - 0.46 * cos(2 * 3.1415926535 * (x) / (window_length - 1) ) ;% ������
 figure(2)
 plot(w,'b');
 title('������')

for i  = 0:range0_end-1
    
        p_start = i * 160;
		p_end = p_start + window_length;
		
		data_line = padlen_input_emphasized_input(p_start+1:p_end);
		 
		data_line = data_line .* w ;% �Ӵ�
  


     data_fft = (abs(fft(data_line)))  ;%�ɴ˵õ���ֵ��С����python�в��
    
   %data_fft = (fft(data_line))  ;%�ɴ˵õ���ֵ��С����python�в��

      data_fft  = (data_fft.*data_fft) ./512;%���������ڼ��㹦����
       
       %data_fft  = (data_fft.*data_fft);%���������ڼ��㹦����

      % data_input(i+1,:,1)=data_line(:); %����Ϊ400����2��ֵ����200����ȡһ�����ݣ���Ϊ�ǶԳƵ�
      data_input(i+1,1:window_length/2,1)=data_fft(1:window_length/2); %����Ϊ400����2��ֵ����200����ȡһ�����ݣ���Ϊ�ǶԳƵ�
 
%VAD%5%%%///////////////////////////////////        
  senergy_amount  = sum(data_fft(1:window_length/2));%����
        if (senergy_jishu_2<4)
        senergy_ths =senergy_ths+senergy_amount;
        senergy_jishu_2=senergy_jishu_2+1;
        end
        if (senergy_jishu_2==4)
            senergy_ths=16*(senergy_ths/4);
            %senergy_ths = 197950000;
            %right��ֵ 197950000
            senergy_jishu_2=32;
        end
        
        
         if (senergy_amount<senergy_ths && senergy_flag_1==0)
                senergy_jishu_1=0;
            end
        if (senergy_amount>senergy_ths && senergy_flag_1==0)
            senergy_jishu_1=senergy_jishu_1+1;
            if (senergy_jishu_1==1)
                x1=i+1;
                senergy_flag_1=1;
                senergy_jishu_1=0;
            end
        end
        
         if (senergy_amount>senergy_ths && senergy_flag_2==0&&senergy_flag_1==1)
                senergy_jishu_1=0;
         end
         if (senergy_amount<senergy_ths && senergy_flag_2==0 &&senergy_flag_1==1)
            senergy_jishu_1=senergy_jishu_1+1;
            if (senergy_jishu_1>1)
                x2=i+1;
                senergy_flag_2=1;
                senergy_jishu_1=0;
            end
        end
        
        senergy(1,i+1)=senergy_amount;
         figure(7)
        plot(senergy);
        axis([0 100 0 max(senergy)]);
         line([x1 x1], [min(senergy),max(senergy)], 'Color' , 'red' );
         line([x2 x2], [min(senergy),max(senergy)], 'Color' , 'red' );
         
%5%%%///////////////////////////////////
         
end

if (senergy_flag_2==1)
            data_input=data_input(x1:x2,1:window_length/2);
end

max(data_input,[],'all')
 min(data_input,[],'all')  
 
            % Compute the mel spectrogram by using the filterbank
 mel_spectrogram = data_input*mel_filterbank_in;%����fbank
 max(mel_spectrogram,[],'all')
 min(mel_spectrogram,[],'all') 
mel_spectrogram_out = log(mel_spectrogram+1);

mel_spectrogram_out = fi(mel_spectrogram_out,0,8,3);
mel_spectrogram_out = mel_spectrogram_out*8;
% mel_spectrogram_out = mel_spectrogram_out(1:65,:);

figure(6)
image(mel_spectrogram'*0.01);
title('FbankƵ��')

% q = quantizer('single');
% aaa = num2bin(q,mel_spectrogram_out(1,1));

% fid1=fopen('G:\peprocess\model_matlab\mel_spectrogram_out.coe','w+');
% fprintf(fid1,'memory_initialization_radix=2;');
% fprintf(fid1,'memory_initialization_vector=');
% for i=1:size(mel_spectrogram_out,1)
%     for j=1:size(mel_spectrogram_out,2)
%         q = quantizer('single');
%         aaa = num2bin(q,mel_spectrogram_out(i,j));
%         fprintf(fid1,'%c',aaa);
%         fprintf(fid1,',\n');
%     end
% end
% fclose(fid1);



fid1=fopen('G:\peprocess\model_matlab\mel_spectrogram_out.coe','w+');
% fprintf(fid1,'memory_initialization_radix=10;');
% fprintf(fid1,'memory_initialization_vector=');
for i=1:size(mel_spectrogram_out,1)
    for j=1:size(mel_spectrogram_out,2)
        fprintf(fid1,'%d',mel_spectrogram_out(i,j));
        fprintf(fid1,',');
    end
end
fclose(fid1);


   
   