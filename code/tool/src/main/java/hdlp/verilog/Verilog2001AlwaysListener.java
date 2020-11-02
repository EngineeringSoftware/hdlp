package hdlp.verilog;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;

import java.io.Writer;
import java.io.IOException;

/**
 * Listener to collect always statement.
 */
public class Verilog2001AlwaysListener extends Verilog2001BaseListener {

    private Writer outFile;

    public Verilog2001AlwaysListener(Writer o) {
        outFile = o;
    }
    
    private void alwaysPrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void enterAlways_construct(Verilog2001Parser.Always_constructContext ctx) {
        int line_num = ctx.start.getLine();
        alwaysPrintHelper("<always> "+String.valueOf(line_num)+"\n");
    }
}
