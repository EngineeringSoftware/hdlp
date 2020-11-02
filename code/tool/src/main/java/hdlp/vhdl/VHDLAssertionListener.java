package hdlp.vhdl;


import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.misc.Interval;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

/**
 * Listener to collect process statement.
 */
public class VHDLAssertionListener extends vhdlBaseListener {

    private Writer outFile;
    
    public VHDLAssertionListener(Writer o) {
        outFile = o;
    }
    
    private void assertionPrintWriter(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getAssertionTokens(ParseTree q) {
        for (int i = 0; i < q.getChildCount(); i++ ) {
            getAssertionTokens(q.getChild(i));
        }
        if (q.getChildCount() == 0) {
            String toPrint = q.getText();
            if(q.getText().length() > 1) {
                String s[] = q.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            if (!toPrint.equals("")) {
                // assertionTokens.add(toPrint.toLowerCase());
                assertionPrintWriter(toPrint);
                assertionPrintWriter(" ");
            }
        }
    }

    @Override
    public void exitAssertion_statement(vhdlParser.Assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertionPrintWriter("<assert> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertionTokens(ctx.getChild(i));
        }
        assertionPrintWriter("</assert>\n");
    }

    @Override
    public void exitConcurrent_assertion_statement(vhdlParser.Concurrent_assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertionPrintWriter("<concurrent_assert> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertionTokens(ctx.getChild(i));
        }
        assertionPrintWriter("</concurrent_assert>\n");
    }
    
}
