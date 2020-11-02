package hdlp.verilog;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;

import java.io.Writer;
import java.io.IOException;

/**
 * Listener to collect functions.
 */
public class Verilog2001FunctionListener extends Verilog2001BaseListener {

    private Writer outFile;

    public Verilog2001FunctionListener(Writer o) {
        outFile = o;
    }
    
    private void functionPrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void enterFunction_declaration(Verilog2001Parser.Function_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        functionPrintHelper("<function> "+String.valueOf(line_num)+"\n");
    }
}
