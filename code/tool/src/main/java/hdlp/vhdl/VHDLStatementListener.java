package hdlp.vhdl;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.misc.Utils;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.io.Writer;
import java.io.IOException;

import java.util.List;
import java.util.Queue;
import java.util.LinkedList;
import java.util.ArrayList;

/**
 * Visitor that finds all concurrent statements.
 */
public class VHDLStatementListener extends vhdlBaseListener {

    private Writer outFile;
    private List<String> tokens;
    private int MAX_AGN_LENGTH;
    
    public VHDLStatementListener(Writer o) {
        outFile = o;
        tokens = new ArrayList<String>();
        MAX_AGN_LENGTH = 200;
    }

    private void getTokens(ParseTree q) {
        for (int i = 0; i < q.getChildCount(); i++ ) {
            getTokens(q.getChild(i));
        }
        if (q.getChildCount() == 0) {
            String toPrint = q.getText();
            if(q.getText().length() > 1) {
                String s[] = q.getText().split(";");
                if(s.length > 1) {
                    toPrint = s[s.length-1];
                }
            }
            if (!toPrint.equals("")) {
                tokens.add(toPrint.toLowerCase());
            }
        }
    }

    private boolean checkWhenKeyword() {
        int j = tokens.indexOf("<=");
        for (int i = j; i<tokens.size(); i++) {
            if (tokens.get(i).equals("when")) {
                return true;
            }
        }
        return false;
    }
    
    // @Override
    // public void exitConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
    //     try {
    //         for (int i = 0; i < ctx.getChildCount(); i++ ) {
    //             getTokens(ctx.getChild(i));
    //         }
    //         boolean check = checkWhenKeyword();
    //         if (check) {
                
    //             if (!(tokens.size()>MAX_AGN_LENGTH)) {
    //                 outFile.write("<signal_cond> ");
    //                 for (int i = 0; i<tokens.size(); i++) {
    //                     outFile.write(tokens.get(i));
    //                     outFile.write(" ");
    //                 }
    //                 outFile.write("\n");
    //             }
    //         } else {
    //             if (!(tokens.size()>MAX_AGN_LENGTH)) {
    //                 outFile.write("<signal_conc> ");
    //                 for (int i = 0; i<tokens.size(); i++) {
    //                     outFile.write(tokens.get(i));
    //                     outFile.write(" ");
    //                 }
    //                 // outFile.write(String.valueOf(tokens.size()));
    //                 outFile.write("\n");
    //             }
    //         }
    //         tokens.clear();
    //     } catch (IOException ex) {
    //         ex.printStackTrace();
    //     }
    // }

    @Override public void exitSelected_signal_assignment(vhdlParser.Selected_signal_assignmentContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++ ) {
                getTokens(ctx.getChild(i));
            }
            if (tokens.size()<=MAX_AGN_LENGTH) {
                outFile.write("<signal_sel>\n");
            } else {
                outFile.write("<too_long_signal_sel> "+String.valueOf(tokens.size())+" <length>\n");
            }
            tokens.clear();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitConditional_signal_assignment(vhdlParser.Conditional_signal_assignmentContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++ ) {
                getTokens(ctx.getChild(i));
            }
            boolean check = checkWhenKeyword();
            if (check) {
                if (tokens.size()<=MAX_AGN_LENGTH) {
                    outFile.write("<signal_cond> ");
                    for (int i = 0; i<tokens.size(); i++) {
                        outFile.write(tokens.get(i));
                        outFile.write(" ");
                    }
                    outFile.write("\n");
                } else {
                    outFile.write("<too_long_signal_cond> "+String.valueOf(tokens.size())+" <length>\n");
                }
            } else {
                if (tokens.size()<=MAX_AGN_LENGTH) {
                    outFile.write("<signal_conc> ");
                    for (int i = 0; i<tokens.size(); i++) {
                        outFile.write(tokens.get(i));
                        outFile.write(" ");
                    }
                    outFile.write("\n");
                } else {
                    outFile.write("<too_long_signal_conc> "+String.valueOf(tokens.size())+" <length>\n");
                }
            }
            tokens.clear();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }


    @Override public void exitSignal_assignment_statement(vhdlParser.Signal_assignment_statementContext ctx) {
        try {
            for (int i = 0; i < ctx.getChildCount(); i++ ) {
                getTokens(ctx.getChild(i));
            }
            boolean check = checkWhenKeyword();
            if (check) {
                if (tokens.size()<=MAX_AGN_LENGTH) {
                    
                    outFile.write("<signal_seq> ");
                    for (int i = 0; i<tokens.size(); i++) {
                        outFile.write(tokens.get(i));
                        outFile.write(" ");
                    }
                    outFile.write("\n");
                }
            }
            tokens.clear();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitComponent_instantiation_statement(vhdlParser.Component_instantiation_statementContext ctx) {
        try {
            
            outFile.write("<component_instantiation>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitAssertion_statement(vhdlParser.Assertion_statementContext ctx) {
        try {                 
            outFile.write("<assertion>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitIf_statement(vhdlParser.If_statementContext ctx) {
        try {                 
            outFile.write("<if>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitGenerate_statement(vhdlParser.Generate_statementContext ctx) {
        try {                 
            outFile.write("<generate>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitLoop_statement(vhdlParser.Loop_statementContext ctx) {
        try {                 
            outFile.write("<loop>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitCase_statement(vhdlParser.Case_statementContext ctx) {
        try {                 
            outFile.write("<case>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitFunction_specification(vhdlParser.Function_specificationContext ctx) {
        try {                 
            outFile.write("<function_call>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitConcurrent_assertion_statement(vhdlParser.Concurrent_assertion_statementContext ctx) {
        try {                 
            outFile.write("<assertion_conc>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitVariable_assignment_statement(vhdlParser.Variable_assignment_statementContext ctx) {
        try {                 
            outFile.write("<variable_assignment>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitProcess_statement(vhdlParser.Process_statementContext ctx) {
        try {                 
            outFile.write("<process>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitBlock_statement(vhdlParser.Block_statementContext ctx) {
        try {                 
            outFile.write("<block>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
}
