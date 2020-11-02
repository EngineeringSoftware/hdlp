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
import java.util.LinkedList;
import java.util.Collections;

public class VHDLContextAssignParallelListener extends vhdlBaseListener {

    private List<String> resultTokens;
    private List<String> assignTokens;
    private List<String> pcontextTokens;
    private List<String> prevAssignTokens;
    private List<List<String>> prevAssignTokensList;
    private List<String> prevAssignClasses;
    private List<String> assignClasses;
    private List<String> generateStmContexts;
    private List<String> blockStmContexts;
    private int prevAssignLength;
    private int maxSeqLen;
    private Writer outFile;

    public VHDLContextAssignParallelListener(Writer o) {
        resultTokens = new ArrayList<String>();
        assignTokens = new ArrayList<String>();
        pcontextTokens = new ArrayList<String>();
        prevAssignTokens = new ArrayList<String>();
        prevAssignTokensList = new ArrayList<List<String>>();
        prevAssignClasses = new ArrayList<String>();
        assignClasses = new ArrayList<String>();
        generateStmContexts = new ArrayList<String>(Arrays.asList("Label_colonContext", "Generation_schemeContext", "TerminalNodeImpl"));
        blockStmContexts = new ArrayList<String>(Arrays.asList("Label_colonContext", "Block_headerContext", "Block_declarative_partContext", "TerminalNodeImpl"));
        prevAssignLength = 0;
        maxSeqLen = 100;
        outFile = o;
    }

    private void writeTokens(List<String> tokens) {
        try {
            for(int i=0; i<tokens.size(); i++) {
                outFile.write(tokens.get(i));
                outFile.write(" ");
            }
            outFile.write("\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    private void assignPrintHelper(ParseTree pt) {
        for (int i = 0; i < pt.getChildCount(); i++ ) {
            assignPrintHelper(pt.getChild(i));
        }
        if (pt.getChildCount() == 0) {
            String toPrint = pt.getText();
            if(pt.getText().length() > 1) {
                String s[] = pt.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            assignTokens.add(toPrint);
            prevAssignTokens.add(toPrint);
        }
    }
    /*
    private void pcontextPrintHelper(ParseTree pt, List<String> temp, boolean startWrite) {
        for (int i = 0; i < pt.getChildCount(); i++ ) {
            pcontextPrintHelper(pt.getChild(i), temp, startWrite);
        }
        if (pt.getChildCount() == 0) {
            String toPrint = pt.getText();
            if(pt.getText().length() > 1) {
                String s[] = pt.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            if (startWrite) {
                pcontextTokens.add(toPrint);
            } else {
                temp.add(toPrint);
            }
        }
        
    }

    private void pcontextExtractor(ParseTree pt, List<String> temp, boolean startWrite) {
        pcontextPrintHelper(pt.getParent(), temp, false);
        if (pt.getParent().getParent().getClass().getSimpleName().equalsIgnoreCase(vhdlParser.Architecture_bodyContext.class.getSimpleName()) && pt.getParent().getClass().getSimpleName().equalsIgnoreCase(vhdlParser.Architecture_statement_partContext.class.getSimpleName())) {
            // || temp.size()>maxSeqLen) {
            startWrite = true;
            int index = pt.getParent().getChildCount()-1;
            pcontextPrintHelper(pt.getParent().getChild(index), temp, startWrite);
            System.out.println(assignTokens);
            
            System.out.println(pt.getParent().getChild(index).getClass().getSimpleName());
            System.out.println(pt.getParent().getChild(index).getText());
            System.out.println("@@@@@@@@@@@@@@@@@@@@@@");
            int indexindex = pt.getParent().getChild(index).getChild(0).getChildCount();
            for (int i=0; i< indexindex; i++) {
                System.out.println(pt.getChild(0).getChild(i).getClass().getSimpleName());
                System.out.println(pt.getChild(0).getChild(i).getText());
                System.out.println(pt.getChild(0).getChild(i).getChildCount());
                if (pt.getChild(0).getChild(i).getChildCount()==1) {
                    System.out.println(pt.getChild(0).getChild(i).getChild(0).getClass().getSimpleName());
                    System.out.println(pt.getChild(0).getChild(i).getChild(0).getText());
                }
                System.out.println("--------------");
            }
            System.out.println("\n\n");
            int contextToIndex = pcontextTokens.size() - assignTokens.size();
            pcontextTokens.add(contextToIndex, "<context>");
        }
        else {
            temp.clear();
            pcontextExtractor(pt.getParent(), temp,  startWrite);
        }
    }
    */
    private void pcontextPrintHelper(ParseTree pt, List<String> temp) {
        for (int i = 0; i < pt.getChildCount(); i++ ) {
            pcontextPrintHelper(pt.getChild(i), temp);
        }
        if (pt.getChildCount() == 0) {
            String toPrint = pt.getText();
            if(pt.getText().length() > 1) {
                String s[] = pt.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            temp.add(toPrint);
        }
    }

    private void pcontextExtractor(ParseTree pt, List<String> temp) {
        if (pt.getParent().getParent().getClass().getSimpleName().equalsIgnoreCase(vhdlParser.Architecture_bodyContext.class.getSimpleName())
            && pt.getParent().getClass().getSimpleName().equalsIgnoreCase(vhdlParser.Architecture_statement_partContext.class.getSimpleName())) {
            pcontextTokens.add("<context>");
        } else if (pt.getClass().getSimpleName().equals(vhdlParser.Block_statementContext.class.getSimpleName())) {
            for (int i = 0; i < pt.getChildCount(); i++) {
                String context = pt.getChild(i).getClass().getSimpleName();
                if (blockStmContexts.contains(context)) {
                    pcontextPrintHelper(pt.getChild(i), temp);
                    //Collections.reverse(temp);
                    //pcontextTokens.addAll(temp);
                    //temp.clear();
                }
            }
            pcontextTokens.addAll(0, temp);
            temp.clear();
            pcontextExtractor(pt.getParent(), temp);
        } else if (pt.getClass().getSimpleName().equalsIgnoreCase(vhdlParser.Generate_statementContext.class.getSimpleName())) {
            for (int i = 0; i < pt.getChildCount(); i++) {
                String context = pt.getChild(i).getClass().getSimpleName();
                if (generateStmContexts.contains(context)) {
                    pcontextPrintHelper(pt.getChild(i), temp);
                    //Collections.reverse(temp);
                    //pcontextTokens.addAll(temp);
                    //temp.clear();
                }
            }
            pcontextTokens.addAll(0, temp);
            temp.clear();
            pcontextExtractor(pt.getParent(), temp);
        } else {
            temp.clear();
            pcontextExtractor(pt.getParent(), temp);
        }
    }
    
        
    @Override
    public void exitConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
        
        // add new concurrent signal assignment tokens
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            assignPrintHelper(ctx.getChild(i));
        }
        String pClass = ctx.toString();
        
        List<String> temp = new ArrayList<String>();
        pcontextExtractor(ctx.getParent(), temp);
        if(prevAssignLength>0) {
            prevAssignTokens.subList(0, prevAssignLength).clear();
            int lens = prevAssignTokensList.size();
            boolean isFirstPrevAssignment = true;
            for (int i = 0; i < lens; i++) {
                List<String> prevAssignObj = prevAssignTokensList.get(i);
                if (prevAssignClasses.get(i).equalsIgnoreCase(pClass)) {
                    if (isFirstPrevAssignment) {
                        isFirstPrevAssignment = false;
                    } else {
                        resultTokens.add("<prevassign>");
                    }
                    resultTokens.addAll(prevAssignObj);
                }
            }
            resultTokens.add("<prevassign>");
            resultTokens.addAll(pcontextTokens);
            resultTokens.addAll(assignTokens);
            writeTokens(resultTokens);
            resultTokens.clear();
        } else {
            resultTokens.add("<prevassign>");
            resultTokens.addAll(pcontextTokens);
            resultTokens.addAll(assignTokens);
            writeTokens(resultTokens);
            resultTokens.clear();
        }
        List<String> prevAssignTokensObj = new ArrayList<String>(prevAssignTokens);
        prevAssignTokensList.add(prevAssignTokensObj);
        prevAssignClasses.add(pClass);
        prevAssignLength = prevAssignTokens.size();
        assignTokens.clear();
        pcontextTokens.clear();
    }
    
    @Override
    public void exitSignal_assignment_statement(vhdlParser.Signal_assignment_statementContext ctx){

    }
}
