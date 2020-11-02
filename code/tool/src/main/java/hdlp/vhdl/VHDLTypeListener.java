package hdlp.vhdl;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.commons.text.StringEscapeUtils;

import java.io.Writer;
import java.io.IOException;


/**
 * Listener to finds all identifier types and initialization.
 */

public class VHDLTypeListener extends vhdlBaseListener {

    private Writer outFile;
    private boolean flag;
    private String CONSTANT = "constant";
    private String SIGNAL = "signal";
    private String VARIABLE = "variable";
    private String ATTRIBUTE = "attribute";
    private String TYPE = "type";
    private String SUBTYPE = "subtype";
    private String ALIAS = "alias";

    
    public VHDLTypeListener(Writer o) {
        outFile = o;
        flag = false;
    }

    private void treePrintHelper(ParseTree q, String syntaxWord) {
        try {
            for (int i = 0; i < q.getChildCount(); i++ ) {
                treePrintHelper(q.getChild(i), syntaxWord);
            }
            if (q.getChildCount() == 0) {
                String toPrint = q.getText();
                if (!toPrint.equals(";") && !toPrint.toLowerCase().equals(syntaxWord)) {
                    outFile.write(StringEscapeUtils.escapeJson(toPrint));
                    outFile.write(" ");
                }
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitSignal_declaration(vhdlParser.Signal_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), SIGNAL);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitAttribute_declaration(vhdlParser.Attribute_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), ATTRIBUTE);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    
    @Override
    public void exitVariable_declaration(vhdlParser.Variable_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), VARIABLE);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitConstant_declaration(vhdlParser.Constant_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), CONSTANT);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitAlias_declaration(vhdlParser.Alias_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), ALIAS);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitType_declaration(vhdlParser.Type_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), TYPE);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void exitSubtype_declaration(vhdlParser.Subtype_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), SUBTYPE);
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    @Override
    public void exitInterface_port_declaration(vhdlParser.Interface_port_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), "");
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }

    }

    @Override
    public void exitInterface_constant_declaration(vhdlParser.Interface_constant_declarationContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++) {
                treePrintHelper(ctx.getChild(i), "");
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
}
