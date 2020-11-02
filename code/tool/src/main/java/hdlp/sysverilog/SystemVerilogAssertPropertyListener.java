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
 * Visitor that finds all Assert_Property statement.
 */
public class SystemVerilogAssertPropertyListener extends SystemVerilogBaseListener {

    private Writer outFile;
    
    public SystemVerilogAssertPropertyListener(Writer o) {
        // assertPropertyTokens = new ArrayList<String>();
        outFile = o;
    }

    private void assertPropertyWriter(String ent) {
        try {
            outFile.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getAssertPropertyTokens(ParseTree q) {
        for (int i = 0; i < q.getChildCount(); i++ ) {
            getAssertPropertyTokens(q.getChild(i));
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
                assertPropertyWriter(toPrint);
                assertPropertyWriter(" ");
            }
            // assertPropertyTokens.add(toPrint.toLowerCase());
        }
    }
    
    @Override
    public void exitAssert_property_statement(SystemVerilogParser.Assert_property_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<assert_property> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
        }
        assertPropertyWriter("\n");
    }

    @Override public void exitProperty_declaration(SystemVerilogParser.Property_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<property_declaration> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
        }
        assertPropertyWriter("\n");
    }

    @Override
    public void exitConcurrent_assertion_item(SystemVerilogParser.Concurrent_assertion_itemContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<concurrent_assertion> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
        }
        assertPropertyWriter("\n");
    }

    @Override
    public void exitProcedural_assertion_statement(SystemVerilogParser.Procedural_assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<procedural_assertion> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
        }
        assertPropertyWriter("\n");
    }

    @Override
    public void exitImmediate_assertion_statement(SystemVerilogParser.Immediate_assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<immediate_assertion> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
            assertPropertyWriter("\n");       
        }
    }

    @Override public void exitSimple_immediate_assertion_statement(SystemVerilogParser.Simple_immediate_assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<simple_immediate_assertion> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
            assertPropertyWriter("\n");
        }
    }

    @Override public void exitSimple_immediate_assert_statement(SystemVerilogParser.Simple_immediate_assert_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<simple_immediate_assert> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
            assertPropertyWriter("\n");
        }
    }

    @Override public void exitDeferred_immediate_assertion_statement(SystemVerilogParser.Deferred_immediate_assertion_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<deferred_immediate_assertion> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
            assertPropertyWriter("\n");
        }
    }

    @Override public void exitDeferred_immediate_assert_statement(SystemVerilogParser.Deferred_immediate_assert_statementContext ctx) {
        int line_num = ctx.start.getLine();
        assertPropertyWriter("<deferred_immediate_assert> "+String.valueOf(line_num)+" ");
        for (int i = 0; i < ctx.getChildCount(); i++ ) {
            getAssertPropertyTokens(ctx.getChild(i));
            assertPropertyWriter("\n");
        }
    }
}
