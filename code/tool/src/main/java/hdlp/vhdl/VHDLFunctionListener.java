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
 * Listener to collect functions.
 */
public class VHDLFunctionListener extends vhdlBaseListener {
    
    private int line_num;
    private Writer outFile;
    private List<String> functionTokens;
    private List<String> functionName;

    public VHDLFunctionListener(Writer o) {
        line_num = -1;
        outFile = o;
        functionTokens = new ArrayList<String>();
        functionName = new ArrayList<String>();
    }
    
    private void functionPrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getFunctionTokens(ParseTree q) {
        for (int i = 0; i < q.getChildCount(); i++ ) {
            getFunctionTokens(q.getChild(i));
        }
        if (q.getChildCount() == 0) {
            String toPrint = q.getText();
            if(q.getText().length() > 1) {
                String s[] = q.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            functionTokens.add(toPrint.toLowerCase());
        }
    }
    
    private void getFunctionName() {
        int fnIndexFrom = functionTokens.indexOf("function");
        int fnReturnIndex = functionTokens.indexOf("return");
        int fnBracketIndex = functionTokens.indexOf("(");
        int fnIndexTo = -1;
        if (fnBracketIndex>0 && fnBracketIndex<fnReturnIndex) {
            fnIndexTo = fnBracketIndex;
        } else if (fnBracketIndex<0 && fnReturnIndex>0) {
            fnIndexTo = fnReturnIndex;
        }
        
        if (fnIndexFrom < fnIndexTo) {
            functionName = functionTokens.subList(fnIndexFrom+1, fnIndexTo);
        } else {
            functionName = functionTokens.subList(fnIndexFrom, fnIndexFrom+1);
        }
    }
    
    @Override
    public void exitFunction_specification(vhdlParser.Function_specificationContext ctx) {
        line_num = ctx.start.getLine();
        functionPrintHelper("<function> "+String.valueOf(line_num));

        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getFunctionTokens(ctx.getChild(i));
        }
        getFunctionName();
        for (int i = 0; i<functionName.size(); i++) {
            if (!functionName.get(i).equalsIgnoreCase("function")) {
                functionPrintHelper(" "+functionName.get(i));
            }
        }
        functionPrintHelper(" </function>\n");
        functionName.clear();
        functionTokens.clear();
    }
}
