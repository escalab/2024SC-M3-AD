clc; clear;

rng(0);
W = 12;
test_case = 20;
FP = fimath('OverflowAction', 'Wrap');
% Pattern Files
op_in = fopen('Pattern/op_in.dat', 'w');
data_in = fopen('Pattern/data_in.dat', 'w');
data_out = fopen('Pattern/gold_data_out.dat', 'w');

for idx = 1:test_case
    mode = randi([0 2]);
    if(mode == 0)
        inA = randi([0, 2^W-1], 4, 4);
        inB = randi([0, 2^W-1], 4, 4);
        out = inA * inB;
        out_fi = fi(out, 0, 2*W, 0, FP);
        out = double(out_fi);
        
        fprintf(op_in,'%s\n', '0');
        inA_hex = concat(inA, W);
        inB_hex = concat(inB, W);
        fprintf(data_in,'%s\n', inA_hex);
        fprintf(data_in,'%s\n', inB_hex);
        out_hex = concat(out, 2*W);
        fprintf(data_out,'%s\n', out_hex);
    elseif(mode == 1)
        inA = randi([0, 2^(2*W)-1], 2, 4);
        inB = randi([0, 2^(2*W)-1], 4, 2);
        out = inA * inB;
        
        out_fi = fi(out, 0, 4*W, 0, FP);
        out = double(out_fi);
        out = [out; zeros(2)];
        
        fprintf(op_in,'%s\n', '2');
        inA_hex = concat(inA, 2*W);
        inB_hex = concat(inB, 2*W);
        fprintf(data_in,'%s\n', inA_hex);
        fprintf(data_in,'%s\n', inB_hex);
        out_hex = concat(out, 4*W);
        fprintf(data_out,'%s\n', out_hex);
    else
        inA = randi([0, 2^(2*W)-1], 4, 2);
        inB = randi([0, 2^(2*W)-1], 2, 4);
        out = inA * inB;
        out_fi = fi(out, 0, 4*W, 0, FP);
        out = double(out_fi);
        
        fprintf(op_in,'%s\n', '3');
        inA_hex = concat(inA, 2*W);
        inB_hex = concat(inB, 2*W);
        fprintf(data_in,'%s\n', inA_hex);
        fprintf(data_in,'%s\n', inB_hex);
        out_hex = concat(out(1:2, :), 4*W);
        fprintf(data_out,'%s\n', out_hex);
        out_hex = concat(out(3:4, :), 4*W);
        fprintf(data_out,'%s\n', out_hex);
    end
end

fclose(op_in);
fclose(data_in);
fclose(data_out);

function [concat_hex] = concat(mat_in, digits)
    [n_row, n_col] = size(mat_in);
    bin_row = [];
    for r_idx = 1:n_row
        for c_idx = 1:n_col
            bin_row = [dec2bin(mat_in(r_idx, c_idx), digits) bin_row];
        end
    end
    hex_row = cell2mat(regexprep(cellstr(reshape(bin_row,4,[]).'), cellstr(dec2bin(0:15,4)), cellstr(dec2hex(0:15)),'once').');
    concat_hex = lower(hex_row);
end

