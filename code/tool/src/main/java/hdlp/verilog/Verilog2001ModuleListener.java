package hdlp.verilog;

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
 * Visitor that finds all instances of module.
 */
public class Verilog2001ModuleListener extends Verilog2001BaseListener {

    private Writer outFile;
    // private Writer outFile1;
    // private Writer outFile2;
    // private List<String> entity_tokens;
    
    public Verilog2001ModuleListener(Writer o) {
        // entity_tokens = new ArrayList<String>();
        outFile = o;
    }

    private void modulePrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void enterModule_declaration(Verilog2001Parser.Module_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        modulePrintHelper("<module> "+String.valueOf(line_num)+"\n");
        // for (int i = 0; i < ctx.getChildCount(); i++ ) {
        //     treePrintHelper(ctx.getChild(i));
        // }
    }    
}
