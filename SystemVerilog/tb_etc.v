`timescale 1ns/10ps
`define CYCLE 10.0

module tb_etc;

    localparam INPUT_DELAY  = 0.5;
    localparam OUTPUT_DELAY = 0.5;
    localparam TEST_CASE    = 20;  
    localparam LATENCY      = 2;

    localparam W            = 12;
    
    reg                     clk;
    // Module in/out ports
    reg           [    3:0] tb_op;
    reg [3:0][3:0][  W-1:0] tb_inA, tb_inB;
    reg [3:0][3:0][2*W-1:0] tb_out;

    // Golden Truth
    reg [3:0][3:0][2*W-1:0] gold_out;

    // Some controls
    reg                     in_stall;
    reg [LATENCY-2:0][ 3:0] op_reg;
    reg              [3 :0] out_op;
    reg                     op3_flag;
    
    //
    reg               [3:0] pat_op;
    reg         [4*4*W-1:0] pat_data;
    reg       [4*4*2*W-1:0] gold_pat;

    // Counters
    integer test_in_idx, test_out_idx;
    integer tot_err;

    // For loop
    integer for_i, for_j;

    // For pattern files
    integer val_op_in, fp_op_in;
    integer val_data_in, fp_data_in;
    integer val_data_out, fp_data_out;

    task CloseFile; begin
        $fclose(fp_op_in);
        $fclose(fp_data_in);
        $fclose(fp_data_out);
    end
    endtask

    initial begin
        fp_op_in    = $fopen("Pattern/op_in.dat", "r");
        fp_data_in  = $fopen("Pattern/data_in.dat", "r");
        fp_data_out = $fopen("Pattern/gold_data_out.dat", "r");
    end

    always #(`CYCLE*0.5) begin
        clk = ~clk;
    end

    initial begin
        $fsdbDumpfile("./etc.fsdb");
        $fsdbDumpvars(0, tb_etc, "+mda");
    end

    etc #(.W(W)) test_etc
    (
        .clk(clk),
        .op (tb_op),
        .inA(tb_inA),
        .inB(tb_inB),
        .out(tb_out)
    );

    initial begin       
        clk     = 1'b1;
        $display("========= Start =========");
        #(`CYCLE * 1500)
        $display("========= Timeout =========");
        $finish;
    end

    initial begin
        test_in_idx = 0;
        tb_op  = 0;
        tb_inA = 0;
        tb_inB = 0;
        in_stall = 0;

        #(5 * `CYCLE);
        while(test_in_idx < TEST_CASE) begin
            @(posedge clk);
            #(INPUT_DELAY * `CYCLE);
            if(in_stall) begin
                in_stall = 0;
                test_in_idx = test_in_idx + 1;
            end
            else begin
                val_op_in = $fscanf(fp_op_in , "%h", pat_op);
                tb_op = pat_op;

                val_data_in = $fscanf(fp_data_in , "%h", pat_data);
                for (for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    for (for_j = 0; for_j < 4; for_j = for_j + 1) begin
                        tb_inA[for_i][for_j] = pat_data[(4*for_i+for_j+1)*W-1 -: W];
                    end
                end

                val_data_in = $fscanf(fp_data_in , "%h", pat_data);
                for (for_i = 0; for_i < 4; for_i = for_i + 1) begin
                    for (for_j = 0; for_j < 4; for_j = for_j + 1) begin
                        tb_inB[for_i][for_j] = pat_data[(4*for_i+for_j+1)*W-1 -: W];
                    end
                end
                if(tb_op == 4'd3) in_stall = 1;
                else test_in_idx = test_in_idx + 1;
            end
        end
        @(posedge clk);
        #(INPUT_DELAY * `CYCLE);
        tb_op  = 0;
        tb_inA = 0;
        tb_inB = 0;
    end

    always @(posedge clk) begin
        out_op = op_reg[LATENCY-2];
        for (for_i = LATENCY-2; for_i > 0; for_i = for_i + 1) begin
            op_reg[for_i] = op_reg[for_i-1];
        end
        op_reg[0] = tb_op;
    end

    initial begin
        tot_err = 0;
        test_out_idx = 0;
        op3_flag = 0;

        #(5 * `CYCLE);
        #(LATENCY * `CYCLE);
        while(test_out_idx < TEST_CASE) begin
            @(negedge clk);
            val_data_out = $fscanf(fp_data_out , "%h", gold_pat);
            for (for_i = 0; for_i < 4; for_i = for_i + 1) begin
                for (for_j = 0; for_j < 4; for_j = for_j + 1) begin
                    gold_out[for_i][for_j] = gold_pat[(4*for_i+for_j+1)*(2*W)-1 -: 2*W];
                end
            end

            if(op3_flag) $display("Number of Test Case: %d, OP: %s", test_out_idx, " 3 (second cycle)");
            else         $display("Number of Test Case: %d, OP: %d", test_out_idx, out_op);
            for (for_i = 0; for_i < 4; for_i = for_i + 1) begin
                for (for_j = 0; for_j < 4; for_j = for_j + 1) begin
                    if((out_op !== 2) || (for_i < 2)) begin
                        if(tb_out[for_i][for_j] !== gold_out[for_i][for_j]) begin
                    	    $display("Index: (%d, %d)", for_i, for_j);
                    	    $display("Expect Value: %d, Get: %d", 
                    	              gold_out[for_i][for_j], tb_out[for_i][for_j]);
                    	    tot_err = tot_err + 1;
                    	end
                    end
                end
            end

            if((out_op == 3'd3) && (!op3_flag)) begin
                op3_flag = 1;
                test_out_idx = test_out_idx;
            end
            else begin
                op3_flag = 0;
                test_out_idx = test_out_idx + 1;
            end
            
            if(tot_err > 60) begin
                $display("=================================================");
                $display("|     ERRORS OCCUR!! Please check the code!     |");
                $display("=================================================");
                CloseFile;
                $finish;
            end
        end

        @(posedge clk);
        $display("\n");
        $display("=======================The test result is ..... Correct!!=========================");
        $display("\n");
        $display("        *****************************************************                        ");
        $display("        **                                                 **      /|__/|            ");
        $display("        **                                                 **     / O,O  \\          ");
        $display("        **     Congratulations!! Your code is PASSED!!     **    /_____   \\         ");
        $display("        **                                                 **   /^ ^ ^ \\  |         ");
        $display("        **                                                 **  |^ ^ ^ ^ |w|          ");
        $display("        *****************************************************   \\m___m__|_|         ");
        $display("\n");
        $display("=====================================================================================");
        $display("\n");
        CloseFile;
        $finish;
    end

endmodule
    

