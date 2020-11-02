package hdlp.vhdl;


import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RuleContext;
import org.antlr.v4.runtime.ParserRuleContext;
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
 * Listener to collect process statement.
 */
public class VHDLProcessListener extends vhdlBaseListener {

    private int line_num;
    private Writer outFile;
    private List<String> processTokens;
    private List<String> processName;
    private static final String[] blacklistedRules = new String[] {"process_declarative_part", "slice_name_part"};
    private static final int MAX_TOKEN_LENGTH = 50;
                                                          

    public VHDLProcessListener(Writer o) {
        line_num = -1;
        outFile = o;
        processTokens = new ArrayList<String>();
        processName = new ArrayList<String>();
    }
    
    private void processPrintHelper(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getProcessTokens(ParseTree q) {
        for (int i = 0; i < q.getChildCount(); i++ ) {
            getProcessTokens(q.getChild(i));
        }
        if (q.getChildCount() == 0) {
            String toPrint = q.getText();
            if(q.getText().length() > 1) {
                String s[] = q.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            processTokens.add(toPrint.toLowerCase());
        }
    }

    private void getProcessName() {
        int processIndex = processTokens.indexOf("process");
        if (processIndex>0){
            processName = processTokens.subList(0, processIndex+1);
        } else {
            processName = processTokens.subList(0, 1);
        }
    }
    
    private void printASTHelper(RuleContext ctx){
        boolean unignoredChild = false;
        String ruleName = vhdlParser.ruleNames[ctx.getRuleIndex()];
        boolean ignoreElem = (ctx.getChildCount() == 1 && ctx.getChild(0) instanceof ParserRuleContext);
        boolean hasExplorableChildren = false;
        for (int i=0;i<ctx.getChildCount();i++) {
            ParseTree element = ctx.getChild(i);
            if (element instanceof RuleContext) {
                String elemName = vhdlParser.ruleNames[((RuleContext)element).getRuleIndex()];
                boolean blacklisted = Arrays.asList(blacklistedRules).contains(elemName);
                if (!blacklisted) {
                    hasExplorableChildren = true;
                }
            }
        }
        if (!ignoreElem) {
            String text = ctx.getText();
            boolean blacklisted = Arrays.asList(blacklistedRules).contains(ruleName);
            // for (String s : printRuleOnly) {
            //     if (ruleName.contains(s)){
            //         printOnlyRule = true;
            //     }
            // }
            
            if (!blacklisted) {
                if (hasExplorableChildren){
                    processPrintHelper(ruleName+":: ");
                }
                else {
                    processPrintHelper(ruleName+"::"+text.substring(0,Math.min(text.length(),MAX_TOKEN_LENGTH))+" ");
                }
            }
        }
        
        if (!ignoreElem && hasExplorableChildren){
            processPrintHelper("( ");
        }
        for (int i=0;i<ctx.getChildCount();i++) {
            ParseTree element = ctx.getChild(i);
            if (element instanceof RuleContext) {
                printASTHelper((RuleContext)element);
            }
        }
        if (!ignoreElem && hasExplorableChildren){
            processPrintHelper(") ");
        }
    }
    
    @Override
    public void exitProcess_statement(vhdlParser.Process_statementContext ctx) {
        line_num = ctx.start.getLine();
        processPrintHelper("<process> <ln> "+String.valueOf(line_num)+" </ln>");

        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getProcessTokens(ctx.getChild(i));
        }
        getProcessName();
        processPrintHelper(" <name> ");
        for (int i = 0; i<processName.size(); i++) {
            if (!processName.get(i).equals(":")&&!processName.get(i).equalsIgnoreCase("process")) {
                processPrintHelper(" "+processName.get(i));
            }
        }
        processPrintHelper(" </name> <code> ");        
        for (int i = 0; i<processTokens.size(); i++) {
            processPrintHelper(" "+processTokens.get(i));
        }
        processPrintHelper(" </code> <process_ast> ");
        processPrintHelper("( ");
        printASTHelper(ctx);
        processPrintHelper(") ");
        processPrintHelper(" </process_ast> </process>\n");
        processName.clear();
        processTokens.clear();
    }
}
