package hdlp;

import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.ParseException;

import hdlp.vhdl.vhdlParser;
import hdlp.vhdl.vhdlLexer;
import hdlp.vhdl.VHDLContextAssignSeqListener;
import hdlp.vhdl.VHDLContextAssignParallelListener;

import hdlp.util.FileUtil;

/**
 * Checks if files can be parsed by ANTLR grammar.
 */
public class example {

    private static void debug(Object o) {
        debug(o.toString());
    }

    private static void debug(String s) {
        System.out.println(s);
    }
    
    /**
     * Check if hdl files with specific extension in a certain project
     * can be parsed and move them to dstDir.
     */
    private static void checkAllFiles(String inputFilePath, String prevassignType) throws Exception {
        boolean isParsable = checkIfParsable(inputFilePath, prevassignType);
        debug(isParsable);
        debug("Done.");
    }
    
    private static boolean checkIfVhdlParsable(CharStream charStream, String prevassignType) throws IOException {

        FileWriter assignWriter = null;
        assignWriter = new FileWriter("example_output.asg");
        vhdlLexer lexer = new vhdlLexer(charStream);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        vhdlParser parser = new vhdlParser(tokens);
        if (prevassignType.equalsIgnoreCase("seq")) {
            parser.addParseListener(new VHDLContextAssignSeqListener(assignWriter));
        } else {
            parser.addParseListener(new VHDLContextAssignParallelListener(assignWriter));
        }
        ParseTree tree = parser.design_file(); // parse
        String parsedTree = tree.toStringTree(parser);

        if(parsedTree.startsWith("(design_file (")) {
            assignWriter.close();
            return true;
        } else {
            debug("Does not start with design_file; this file cannot be parsed.");
            return false;
        }
    }
    
    private static boolean checkIfParsable(String dataPath, String prevassignType) throws InterruptedException, RuntimeException {
        debug(dataPath + " ...");
        try {
            CharStream charStream = CharStreams.fromFileName(dataPath);
            return checkIfVhdlParsable(charStream, prevassignType);
        } catch (Exception e) {
            debug(e.getClass());
            e.printStackTrace();
        }
        return false;
    }
    
    public static void main(String[] args) throws Exception {
        Options options = new Options();

        Option input = new Option("i", "inputFile", true, "path of input file");
        Option type = new Option("t", "type", true, "type for extracting previous assignment");
        input.setRequired(true);
        type.setRequired(true);
        options.addOption(input);
        options.addOption(type);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();

        FileWriter fcWriter = null;
        try {
            CommandLine cmd = parser.parse(options, args);
            String inputFilePath = cmd.getOptionValue("inputFile");
            String prevassignType = cmd.getOptionValue("type");
            checkAllFiles(inputFilePath, prevassignType);

        } catch (IOException ex) {
            ex.printStackTrace();
            System.exit(1);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("java example -i <input file path> -t <type for prevassign>", options);
            System.exit(1);
        } finally {
            FileUtil.closeWithoutException(fcWriter);
        }
        System.exit(0);
    }
}
