package hdlp.sysverilog;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

/**
 * Listener to collect output ports.
 */
public class SystemVerilogOutputPortListener extends SystemVerilogBaseListener {

    private Writer outFile;
 
    public SystemVerilogOutputPortListener(Writer o) {
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
    public void exitOutput_declaration(SystemVerilogParser.Output_declarationContext ctx) {
        try {
            ParseTree lastChildTree = ctx.getChild(ctx.getChildCount()-1);
            for (int i = 0; i < lastChildTree.getChildCount(); i++ ) {
                treePrintHelper(lastChildTree.getChild(i));
                outFile.write("\n");
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void exitPort_direction(SystemVerilogParser.Port_directionContext ctx) {
        if (ctx.getText().equals("output")) {
            try {
                outFile.write("<output>\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        
        super.exitPort_direction(ctx);
    }
}
