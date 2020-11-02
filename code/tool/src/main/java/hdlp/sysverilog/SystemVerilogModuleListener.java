package hdlp.sysverilog;

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
 * Visitor that finds all modules.
 */
public class SystemVerilogModuleListener extends SystemVerilogBaseListener {

    private Writer outFile;
   
    public SystemVerilogModuleListener(Writer o) {
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
    public void enterModule_declaration(SystemVerilogParser.Module_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        modulePrintHelper("<module> " + String.valueOf(line_num) + "\n");
    }

    @Override
    public void enterInterface_declaration(SystemVerilogParser.Interface_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        modulePrintHelper("<interface> " + String.valueOf(line_num) + "\n");
    }
}
