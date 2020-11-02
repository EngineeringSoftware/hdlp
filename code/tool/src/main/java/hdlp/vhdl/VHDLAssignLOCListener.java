package hdlp.vhdl;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

import java.util.List;
import java.util.Queue;
import java.util.LinkedList;

/**
 * Visitor that finds all instances of non-blocking assignments.
 */
public class VHDLAssignLOCListener extends vhdlBaseListener {

    private Writer outFile;
    private int loc_enter;
    private int loc_exit;
    
    public VHDLAssignLOCListener(Writer o) {
        outFile = o;
        loc_enter = -1;
        loc_exit = -1;
    }

    private void locPrintHelper(String l) {
        try {
            outFile.write(l);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    // @Override
    // public void enterConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
    //     int loc_enter = ctx.start.getLine();
    // }
    
    @Override
    public void exitConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
        int loc_enter = ctx.getStart().getLine();
        int loc_exit = ctx.getStop().getLine();
        locPrintHelper("<loc> "+String.valueOf(loc_enter)+" "+String.valueOf(loc_exit)+"\n");
        loc_enter = -1;
        loc_exit = -1;
    }
}
