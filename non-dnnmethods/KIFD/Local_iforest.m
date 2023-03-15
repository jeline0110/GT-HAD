function [output_img, stop_flag, s1, index1] = Local_iforest(input_img, data, s, index, lev)  
r0 = input_img;
row = size(input_img,1);
col = size(input_img,2);
stop_flag = 0;
am = im2bw( r0,lev); 
[bw_img,bw_num] = bwlabel(am, 8);
first_flag = 0;
index1 = index;
for i=1:bw_num
    abstract = find(bw_img == i);
    abstract_num = size(abstract,1);
    if abstract_num >= floor(row*col/120)
        first_flag = first_flag + 1;
        if first_flag == 1
            s1 = s;
        end
        TreeData = data(abstract,:);
        NumSub = round(abstract_num * 0.5);
        s2 = iforest(TreeData, 100, NumSub); 
        index_global = abstract;
        index1 = [index1;index_global];
         s2(s2<0.5) = s2(s2<0.5);
        s1(abstract) = s2;
    end
end
if first_flag == 0
stop_flag = 1;
s1 = s;
end
r1 = reshape(s1,row,col);
output_img = r1; 
end