package hdlp.vhdl;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.misc.Interval;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.Queue;
import java.util.LinkedList;

/**
 * Visitor that finds all instances of entity.
 */
public class VHDLComponentListener extends vhdlBaseListener {

    private Writer outFile;
    private List<String> entity_tokens;
    
    public VHDLComponentListener(Writer o) {
        outFile = o;
    }

    private void entityPrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
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
    public void enterComponent_declaration(vhdlParser.Component_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        entityPrintHelper("<component> "+String.valueOf(line_num)+"\n");
        // for (int i = 0; i < ctx.getChildCount(); i++ ) {
        //     treePrintHelper(ctx.getChild(i));
        // }
    }

    @Override
    public void exitComponent_declaration(vhdlParser.Component_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        entityPrintHelper("</component>\n");        
    }
}
