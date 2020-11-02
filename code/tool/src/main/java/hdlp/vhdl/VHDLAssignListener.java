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
public class VHDLAssignListener extends vhdlBaseListener {

    private Writer outFile;
    
    public VHDLAssignListener(Writer o) {
        outFile = o;
    }

    private void treePrintHelper(ParseTree q) {
        try {
            for (int i = 0; i < q.getChildCount(); i++ ) {
                treePrintHelper(q.getChild(i));
            }
            if (q.getChildCount() == 0) {
                String toPrint = q.getText();
                if(q.getText().length() > 1) {
                    String s[] = q.getText().split(";");
                    if(s.length > 1) {
                        toPrint = s[s.length-1];
                    }
                }
                outFile.write(toPrint);
                outFile.write(" ");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void exitConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
        try {                 
            for (int i = 0; i < ctx.getChildCount(); i++ ) {
                treePrintHelper(ctx.getChild(i));
                outFile.write("\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    /*
    @Override
    public void exitSignal_assignment_statement(vhdlParser.Signal_assignment_statementContext ctx){
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i));
            }
            outFile.write("\n");
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    */
}
