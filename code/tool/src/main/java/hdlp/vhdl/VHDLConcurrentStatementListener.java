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

/**
 * Visitor that finds all concurrent statements.
 */
public class VHDLConcurrentStatementListener extends vhdlBaseListener {

    private Writer outFile;
    
    public VHDLConcurrentStatementListener(Writer o) {
        outFile = o;
    }
    
    @Override
    public void exitConcurrent_signal_assignment_statement(vhdlParser.Concurrent_signal_assignment_statementContext ctx) {
        try {                 
            outFile.write("<signal>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitSelected_signal_assignment(vhdlParser.Selected_signal_assignmentContext ctx) {
        try {                 
            outFile.write("<signal>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitConditional_signal_assignment(vhdlParser.Conditional_signal_assignmentContext ctx) {
        try {                 
            outFile.write("<signal>\n");
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

    @Override public void exitComponent_instantiation_statement(vhdlParser.Component_instantiation_statementContext ctx) {
        try {                 
            outFile.write("<component_instantiation>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitConcurrent_assertion_statement(vhdlParser.Concurrent_assertion_statementContext ctx) {
        try {                 
            outFile.write("<assertion>\n");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    @Override public void exitConcurrent_procedure_call_statement(vhdlParser.Concurrent_procedure_call_statementContext ctx) {
        try {                 
            outFile.write("<procedure_call>\n");
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
