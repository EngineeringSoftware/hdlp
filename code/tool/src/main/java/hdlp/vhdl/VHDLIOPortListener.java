package hdlp.vhdl;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

/**
 * Listener to collect input, output and inout ports.
 */
public class VHDLIOPortListener extends vhdlBaseListener {

    private Writer outFile;
    private boolean portDeclarationFlag;
 
    public VHDLIOPortListener(Writer o) {
        outFile = o;
        portDeclarationFlag = false;
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
    public void enterInterface_port_declaration(vhdlParser.Interface_port_declarationContext ctx) {
        portDeclarationFlag = true;
    }
    
    @Override
    public void exitInterface_port_declaration(vhdlParser.Interface_port_declarationContext ctx) {
        portDeclarationFlag = false;
    }

    @Override
    public void exitSignal_mode(vhdlParser.Signal_modeContext ctx) {
        try {
            if (portDeclarationFlag) {
                for (int i = 0; i < ctx.getChildCount(); i++ ) {
                    treePrintHelper(ctx.getChild(i));
                    outFile.write("\n");
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
