`timescale 1ns / 1ps
/*
 * Extended Tensor Cores
 */
 
module etc#(parameter W = 12)
(
    input                      clk,
    input  [3:0]               op,
    input  [3:0][3:0][  W-1:0] inA, inB,
    output [3:0][3:0][2*W-1:0] out
);

    reg       [3:0][3:0][  W-1:0] regA, regB; // input buffer
    reg                 [    3:0] reg_op;
    reg       [3:0][3:0][2*W-1:0] regOut;     // output buffer
    reg                           op3_flag;   // 0 means the first cycle, and 1 means the second cycle
    reg                           op3_flag_d1;

    wire [3:0][3:0][1:0][  2*W:0] mac_out;
    wire      [3:0][3:0][2*W+1:0] add_out;
    wire           [3:0][2*W+2:0] middle_bits_add_out;

    reg            [7:0][2*W+1:0] higher_bits, lower_bits;
    reg            [7:0][2*W+2:0] middle_bits;
    wire           [7:0][4*W-1:0] high_prec_out;
    
    integer                       for_i, for_j;
    genvar                        gen_i, gen_j, gen_k;

    // Units required for different matrix multiplications
    // ========================================================================================================
    // |         Size        | Input Precision | No. of W-bit | No. of 2W-bit         | No. of High-Precision |
    // |                     |                 | Multipliers  | (or (2W+1)-bit) Adder | Adder                 |
    // ========================================================================================================
    // | 4-by-4 times 4-by-4 |        W        |      64      |          48           |           0           |
    // --------------------------------------------------------------------------------------------------------
    // | 2-by-4 times 4-by-2 |       2W        |      64      |          52           |           4           |
    // --------------------------------------------------------------------------------------------------------
    // | 4-by-2 times 2-by-4 |       2W        |      64      |          40           |           8           |
    // --------------------------------------------------------------------------------------------------------
    // where high-precision adder is: (a << 2*W) + (b << W) + c

    // Thus, by the numbers of the block, 64 multipliers, 52 adders, and 8 high-precision adders are deployed in this module.
    // (Without considering to implement the high-precision adders by 2 adders) <- this may reduce the area cost for future optimization

    // 64 multipliers and 32 adders are used here for common operations for three modes
    generate
        for (gen_i = 0; gen_i < 4; gen_i = gen_i + 1) begin
            for (gen_j = 0; gen_j < 4; gen_j = gen_j + 1) begin
                for(gen_k = 0; gen_k < 2; gen_k = gen_k + 1) begin
                    assign mac_out[gen_i][gen_j][gen_k] = regA[gen_i][2*gen_k+0] * regB[2*gen_k+0][gen_j] + regA[gen_i][2*gen_k+1] * regB[2*gen_k+1][gen_j];
                end
            end
        end
    endgenerate

    generate
        for (gen_i = 0; gen_i < 4; gen_i = gen_i + 1) begin
            for (gen_j = 0; gen_j < 4; gen_j = gen_j + 1) begin
                assign add_out[gen_i][gen_j] = mac_out[gen_i][gen_j][0] + mac_out[gen_i][gen_j][1];
            end
        end
    endgenerate
        
    assign middle_bits_add_out[0] = add_out[1][0] + add_out[0][1];
    assign middle_bits_add_out[1] = add_out[1][2] + add_out[0][3];
    assign middle_bits_add_out[2] = add_out[3][0] + add_out[2][1];
    assign middle_bits_add_out[3] = add_out[3][2] + add_out[2][3];

    generate
        for (gen_i = 0; gen_i < 8; gen_i = gen_i + 1) begin
            assign high_prec_out[gen_i] = (higher_bits[gen_i] << 2*W) + (middle_bits[gen_i] << W) + lower_bits[gen_i];
        end
    endgenerate

    assign out = regOut;

    always @(*) begin
        for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
            lower_bits[for_i] = mac_out[0][for_i][0];
            middle_bits[for_i] = add_out[1][for_i];
            higher_bits[for_i] = mac_out[0][for_i][1];
        end
        for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
            lower_bits[4+for_i] = mac_out[2][for_i][0];
            middle_bits[4+for_i] = add_out[3][for_i];
            higher_bits[4+for_i] = mac_out[2][for_i][1];
        end

        case (reg_op[2:0]) // synopsys full_case
            3'd0: begin
                for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    lower_bits[for_i] = mac_out[0][for_i][0];
                    middle_bits[for_i] = add_out[1][for_i];
                    higher_bits[for_i] = mac_out[0][for_i][1];
                end
                for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    lower_bits[4+for_i] = mac_out[2][for_i][0];
                    middle_bits[4+for_i] = add_out[3][for_i];
                    higher_bits[4+for_i] = mac_out[2][for_i][1];
                end
            end
            3'd2: begin
                lower_bits[0] = add_out[0][0];
                middle_bits[0] = middle_bits_add_out[0];
                higher_bits[0] = add_out[1][1];

                lower_bits[1] = add_out[0][2];
                middle_bits[1] = middle_bits_add_out[1];
                higher_bits[1] = add_out[1][3];

                lower_bits[2] = add_out[2][0];
                middle_bits[2] = middle_bits_add_out[2];
                higher_bits[2] = add_out[3][1];

                lower_bits[3] = add_out[2][2];
                middle_bits[3] = middle_bits_add_out[3];
                higher_bits[3] = add_out[3][3];
                for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    lower_bits[4+for_i] = mac_out[2][for_i][0];
                    middle_bits[4+for_i] = add_out[3][for_i];
                    higher_bits[4+for_i] = mac_out[2][for_i][1];
                end
            end
            3'd3: begin
               for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    lower_bits[for_i] = mac_out[0][for_i][0];
                    middle_bits[for_i] = add_out[1][for_i];
                    higher_bits[for_i] = mac_out[0][for_i][1];
                end
                for(for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    lower_bits[4+for_i] = mac_out[2][for_i][0];
                    middle_bits[4+for_i] = add_out[3][for_i];
                    higher_bits[4+for_i] = mac_out[2][for_i][1];
                end
            end
        endcase
    end

    
    always @(posedge clk) begin
    // regOut[16*8-1:0] <= {wireOut[0][7],wireOut[0][6],wireOut[0][5],wireOut[0][4],wireOut[0][3],wireOut[0][2],wireOut[0][1],wireOut[0][0]}};
        reg_op <= op;
        op3_flag_d1 <= op3_flag;
        if((op[2:0] == 3'b000) && (!op3_flag)) begin
            regA[0][0] <= inA[0][0];
            regA[0][1] <= inA[0][1];
            regA[0][2] <= inA[0][2];
            regA[0][3] <= inA[0][3];
            regA[1][0] <= inA[1][0];
            regA[1][1] <= inA[1][1];
            regA[1][2] <= inA[1][2];
            regA[1][3] <= inA[1][3];
            regA[2][0] <= inA[2][0];
            regA[2][1] <= inA[2][1];
            regA[2][2] <= inA[2][2];
            regA[2][3] <= inA[2][3];
            regA[3][0] <= inA[3][0];
            regA[3][1] <= inA[3][1];
            regA[3][2] <= inA[3][2];
            regA[3][3] <= inA[3][3];

            regB[0][0] <= inB[0][0];
            regB[0][1] <= inB[0][1];
            regB[0][2] <= inB[0][2];
            regB[0][3] <= inB[0][3];
            regB[1][0] <= inB[1][0];
            regB[1][1] <= inB[1][1];
            regB[1][2] <= inB[1][2];
            regB[1][3] <= inB[1][3];
            regB[2][0] <= inB[2][0];
            regB[2][1] <= inB[2][1];
            regB[2][2] <= inB[2][2];
            regB[2][3] <= inB[2][3];
            regB[3][0] <= inB[3][0];
            regB[3][1] <= inB[3][1];
            regB[3][2] <= inB[3][2];
            regB[3][3] <= inB[3][3];

        end
        else if((op[2:0] == 3'b010) && (!op3_flag)) begin
            regA[0][0] <= inA[0][0];
            regA[0][1] <= inA[0][2];
            regA[0][2] <= inA[1][0];
            regA[0][3] <= inA[1][2];
            regA[1][0] <= inA[0][1];
            regA[1][1] <= inA[0][3];
            regA[1][2] <= inA[1][1];
            regA[1][3] <= inA[1][3];
            regA[2][0] <= inA[2][0];
            regA[2][1] <= inA[2][2];
            regA[2][2] <= inA[3][0];
            regA[2][3] <= inA[3][2];
            regA[3][0] <= inA[2][1];
            regA[3][1] <= inA[2][3];
            regA[3][2] <= inA[3][1];
            regA[3][3] <= inA[3][3];

            regB[0][0] <= inB[0][0];
            regB[0][1] <= inB[0][1];
            regB[0][2] <= inB[0][2];
            regB[0][3] <= inB[0][3];
            regB[1][0] <= inB[1][0];
            regB[1][1] <= inB[1][1];
            regB[1][2] <= inB[1][2];
            regB[1][3] <= inB[1][3];
            regB[2][0] <= inB[2][0];
            regB[2][1] <= inB[2][1];
            regB[2][2] <= inB[2][2];
            regB[2][3] <= inB[2][3];
            regB[3][0] <= inB[3][0];
            regB[3][1] <= inB[3][1];
            regB[3][2] <= inB[3][2];
            regB[3][3] <= inB[3][3];
        end

        else if((op[2:0] == 3'b011) || op3_flag) begin
            if(op3_flag) begin
                regA[0][0] <= inA[2][0];
                regA[0][1] <= inA[2][2];
                regA[0][2] <= inA[2][1];
                regA[0][3] <= inA[2][3];
                regA[1][0] <= inA[2][1];
                regA[1][1] <= inA[2][3];
                regA[1][2] <= inA[2][0];
                regA[1][3] <= inA[2][2];
                regA[2][0] <= inA[3][0];
                regA[2][1] <= inA[3][2];
                regA[2][2] <= inA[3][1];
                regA[2][3] <= inA[3][3];
                regA[3][0] <= inA[3][1];
                regA[3][1] <= inA[3][3];
                regA[3][2] <= inA[3][0];
                regA[3][3] <= inA[3][2];
            end
            else begin
                regA[0][0] <= inA[0][0];
                regA[0][1] <= inA[0][2];
                regA[0][2] <= inA[0][1];
                regA[0][3] <= inA[0][3];
                regA[1][0] <= inA[0][1];
                regA[1][1] <= inA[0][3];
                regA[1][2] <= inA[0][0];
                regA[1][3] <= inA[0][2];
                regA[2][0] <= inA[1][0];
                regA[2][1] <= inA[1][2];
                regA[2][2] <= inA[1][1];
                regA[2][3] <= inA[1][3];
                regA[3][0] <= inA[1][1];
                regA[3][1] <= inA[1][3];
                regA[3][2] <= inA[1][0];
                regA[3][3] <= inA[1][2];
            end

            regB[0][0] <= inB[0][0];
            regB[0][1] <= inB[0][2];
            regB[0][2] <= inB[1][0];
            regB[0][3] <= inB[1][2];
            regB[1][0] <= inB[2][0];
            regB[1][1] <= inB[2][2];
            regB[1][2] <= inB[3][0];
            regB[1][3] <= inB[3][2];
            regB[2][0] <= inB[0][1];
            regB[2][1] <= inB[0][3];
            regB[2][2] <= inB[1][1];
            regB[2][3] <= inB[1][3];
            regB[3][0] <= inB[2][1];
            regB[3][1] <= inB[2][3];
            regB[3][2] <= inB[3][1];
            regB[3][3] <= inB[3][3];
        end

        if((op[2:0] == 3'b011) && (!op3_flag)) op3_flag = 1;
        else                                   op3_flag = 0;

        if((reg_op[2:0] == 3'b000) && (!op3_flag_d1)) begin
            regOut[0][0] <= add_out[0][0];
            regOut[0][1] <= add_out[0][1];
            regOut[0][2] <= add_out[0][2];
            regOut[0][3] <= add_out[0][3];
            regOut[1][0] <= add_out[1][0];
            regOut[1][1] <= add_out[1][1];
            regOut[1][2] <= add_out[1][2];
            regOut[1][3] <= add_out[1][3];
            regOut[2][0] <= add_out[2][0];
            regOut[2][1] <= add_out[2][1];
            regOut[2][2] <= add_out[2][2];
            regOut[2][3] <= add_out[2][3];
            regOut[3][0] <= add_out[3][0];
            regOut[3][1] <= add_out[3][1];
            regOut[3][2] <= add_out[3][2];
            regOut[3][3] <= add_out[3][3];

        end
        else if((reg_op[2:0] == 3'b010) && (!op3_flag_d1)) begin
            regOut[0][1:0] <= high_prec_out[0];
            regOut[0][3:2] <= high_prec_out[1];
            regOut[1][1:0] <= high_prec_out[2];
            regOut[1][3:2] <= high_prec_out[3];
        end

        else if((reg_op[2:0] == 3'b011) || op3_flag_d1) begin
            regOut[0][1:0] <= high_prec_out[0];
            regOut[0][3:2] <= high_prec_out[1];
            regOut[1][1:0] <= high_prec_out[2];
            regOut[1][3:2] <= high_prec_out[3];
            regOut[2][1:0] <= high_prec_out[4];
            regOut[2][3:2] <= high_prec_out[5];
            regOut[3][1:0] <= high_prec_out[6];
            regOut[3][3:2] <= high_prec_out[7];
        end


    end

endmodule