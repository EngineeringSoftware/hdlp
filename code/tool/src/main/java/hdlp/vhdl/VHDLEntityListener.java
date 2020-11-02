package hdlp.vhdl;

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
 * Visitor that finds all instances of entity.
 */
public class VHDLEntityListener extends vhdlBaseListener {

    private Writer outFile_assign;
    private Writer outFile_typ;
    private Writer outFile_ent;
    // private Writer outFile1;
    // private Writer outFile2;
    // private List<String> entity_tokens;
    
    public VHDLEntityListener(Writer assign_o, Writer typ_o, Writer ent_o) {
        // entity_tokens = new ArrayList<String>();
        outFile_assign = assign_o;
        outFile_typ = typ_o;
        outFile_ent = ent_o;
    }

    private void entityPrintHelper(String ent) {
        try {
            outFile_assign.write(ent);
            outFile_typ.write(ent);
            outFile_ent.write(ent);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    // private void treePrintHelper(ParseTree q) {
    //     try {
    //         for (int i = 0; i < q.getChildCount(); i++ ) {
    //             treePrintHelper(q.getChild(i));
    //         }
    //         if (q.getChildCount() == 0) {
    //             String toPrint = q.getText();
    //             if(q.getText().length() > 1) {
    //                 String s[] = q.getText().split(";");
    //                 if(s.length > 1) {
    //                     toPrint = s[s.length-1];
    //                 }
    //             }
    //             outFile1.write(toPrint);
    //             outFile1.write(" ");
    //             outFile2.write(toPrint);
    //             outFile2.write(" ");
    //         }
    //     } catch (IOException ex) {
    //         ex.printStackTrace();
    //     }
    // }
    
    @Override
    public void enterEntity_declaration(vhdlParser.Entity_declarationContext ctx) {
        int line_num = ctx.start.getLine();
        entityPrintHelper("<entity> "+String.valueOf(line_num)+"\n");
        // for (int i = 0; i < ctx.getChildCount(); i++ ) {
        //     treePrintHelper(ctx.getChild(i));
        // }
    }    
}
