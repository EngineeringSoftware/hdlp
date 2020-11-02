package hdlp.verilog;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

/**
 * Listener to collect input ports.
 */
public class Verilog2001InputPortListener extends Verilog2001BaseListener {

    private Writer outFile;
 
    public Verilog2001InputPortListener(Writer o) {
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
    public void exitInput_declaration(Verilog2001Parser.Input_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++ ) {
                treePrintHelper(ctx.getChild(i));
                outFile.write("\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
