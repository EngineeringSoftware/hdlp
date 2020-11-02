package hdlp;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import hdlp.sysverilog.SystemVerilogSyntaxError;
import hdlp.sysverilog.SystemVerilogSyntaxErrorListener;
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

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.ExecutionException;

import hdlp.verilog.Verilog2001Parser;
import hdlp.verilog.Verilog2001Lexer;
import hdlp.verilog.Verilog2001SyntaxErrorListener;
import hdlp.verilog.Verilog2001SyntaxError;

import hdlp.sysverilog.SystemVerilogLexer;
import hdlp.sysverilog.SystemVerilogParser;

import hdlp.vhdl.vhdlParser;
import hdlp.vhdl.vhdlLexer;
import hdlp.vhdl.vhdlSyntaxErrorListener;
import hdlp.vhdl.vhdlSyntaxError;

import hdlp.util.FileUtil;

/*
 * The following classes is to generate a thread and parse input file
 * on the thread over hdl types.
 */
class VerilogParseTask implements Callable<String> {
    private Verilog2001Parser parser;

    public VerilogParseTask(Verilog2001Parser parser) {
        this.parser = parser;
    }

    public Verilog2001Parser getparser() {
        return parser;
    }

    @Override
    public String call() throws Exception {
        return parser.source_text().toStringTree(parser);
    } 
}

class VhdlParseTask implements Callable<String> {
    private vhdlParser parser;

    public VhdlParseTask(vhdlParser parser) {
        this.parser = parser;
    }

    public vhdlParser getparser() {
        return parser;
    }

    @Override
    public String call() throws Exception {
        return parser.design_file().toStringTree(parser);
    } 
}

class SysVerParseTask implements Callable<String> {
    private SystemVerilogParser parser;

    public SysVerParseTask(SystemVerilogParser parser) {
        this.parser = parser;
    }

    public SystemVerilogParser getparser() {
        return parser;
    }

    @Override
    public String call() throws Exception {
        return parser.system_verilog_text().toStringTree(parser);
    }
}

class World {
    static final String VhdlExt = "vhd";
    static final String VerilogExt = "v";
    static final String SysVerExt = "sv";
    static final String JavaExt = "java";
}

/**
 * Checks if files can be parsed by ANTLR grammar.
 */
public class FileChecker {
    
    private static void debug(Object o) {
        debug(o.toString());
    }

    private static void debug(String s) {
        System.out.println(s);
    }

    /* Copy file at source path to dest path */
    private static void copyFile(File source, File dest) throws IOException {
        FileUtils.copyFile(source, dest);
        debug(source.getName() + " moved to " + dest.getName());
    }

    /* Delete all files in dir directory */
    private static boolean deleteDirectory(File dir) {
        if (dir.isDirectory()) {
            File[] children = dir.listFiles();
            Arrays.sort(children);
            for (int i = 0; i < children.length; i++) {
                boolean success = deleteDirectory(children[i]);
                if (!success) {
                    return false;
                }
            }
        }
        return dir.delete();
    }

    /**
     * Get all files with specific file extension in directory and
     * subdirectories.
     */
    private static void getFiles(String repDir, String plExtension, List<File> files) {
        File fileDir = new File(repDir);
        File[] fList = fileDir.listFiles();
        Arrays.sort(fList);
        if (fList != null) {
            for (File file : fList) {  
                if (file.isFile() && file.toString().endsWith("." + plExtension)) {
                    files.add(file);
                } else if (file.isDirectory()) {
                    getFiles(file.getAbsolutePath(), plExtension, files);
                }
            }
        }
    }
    
    /**
     * Check if hdl files with specific extension in a certain project
     * can be parsed and move them to dstDir.
     */
    private static void checkAllFiles(String projectName, String repoDir, String extension, FileWriter fcWriter) throws Exception {
        List<File> fileList = new ArrayList<File>();
        getFiles(repoDir, extension, fileList);

        int numAvailFiles = 0;
        int numAllFiles = fileList.size();
        
        int i = 0;
        for(File f : fileList) {
            boolean isParsable = checkIfParsable(f.toString(), extension);
            if (isParsable) {
                numAvailFiles++;
            }
            fcWriter.write(f.toString() + ":" + isParsable + "\n");
            i++;
        }
        String result = projectName + ": " + numAvailFiles + " files are available out of " + numAllFiles + " files found\n";
        debug(result);
        debug("Done.");
    }

    private static boolean checkIfVhdlParsable(CharStream charStream, int timeout) throws IOException {
        vhdlLexer lexer = new vhdlLexer(charStream);
        CommonTokenStream stream = new CommonTokenStream(lexer);
        vhdlParser parser = new vhdlParser(stream);

        vhdlSyntaxErrorListener errorListener = new vhdlSyntaxErrorListener();
        parser.addErrorListener(errorListener);
        Callable<String> work = new VhdlParseTask(parser);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(work);
        String parsedTree = "";
        try {
            parsedTree = future.get(timeout, TimeUnit.SECONDS);
            List<vhdlSyntaxError> syntaxErrors = errorListener.getSyntaxErrors();
            if(!syntaxErrors.isEmpty()){
                debug("Syntax Error: This file cannot be parsed.");
                return false;
            }
        } catch (TimeoutException e) {
            future.cancel(true);
            debug("parsing tree timeout!");
            return false;
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();
                    
        if(parsedTree.startsWith("(design_file (")) {
            return true;
        } else {
            debug("Does not start with design_file; this file cannot be parsed.");
            return false;
        }
    }
    
    private static boolean checkIfVerilogParsable(CharStream charStream, int timeout) throws IOException {
        Verilog2001Lexer lexer = new Verilog2001Lexer(charStream);
        lexer.removeErrorListeners();
                    
        CommonTokenStream stream = new CommonTokenStream(lexer);
        Verilog2001Parser parser = new Verilog2001Parser(stream);
        Verilog2001SyntaxErrorListener errorListener = new Verilog2001SyntaxErrorListener();
        parser.addErrorListener(errorListener);
        Callable<String> work = new VerilogParseTask(parser);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(work);
        String parsedTree = "";
        try {
            parsedTree = future.get(timeout, TimeUnit.SECONDS);
            List<Verilog2001SyntaxError> syntaxErrors = errorListener.getSyntaxErrors();
            if (!syntaxErrors.isEmpty()) {
                debug("Syntax Error: This file cannot be parsed.");
                return false;
            }
        } catch (TimeoutException e) {
            future.cancel(true);
            debug("Parsing tree timeout!");
            return false;
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();

        if (parsedTree.startsWith("(source_text (")) {
            return true;
        } else {
            debug("Does not start with source_text; this file cannot be parsed.");
            return false;
        }
    }
    
    private static boolean checkIfSysVerParsable(CharStream charStream, int timeout) throws IOException {
        SystemVerilogLexer lexer = new SystemVerilogLexer(charStream);
        lexer.removeErrorListeners();
        
        CommonTokenStream stream = new CommonTokenStream(lexer);
        SystemVerilogParser parser = new SystemVerilogParser(stream);
        SystemVerilogSyntaxErrorListener errorListener = new SystemVerilogSyntaxErrorListener();
        parser.addErrorListener(errorListener);
        List<SystemVerilogSyntaxError> syntaxErrors = errorListener.getSyntaxErrors();
        
        Callable<String> work = new SysVerParseTask(parser);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(work);
        String parsedTree = "";
        try {
            parsedTree = future.get(timeout, TimeUnit.SECONDS);
            if(!syntaxErrors.isEmpty()){
                debug("Syntax Error: This file cannot be parsed.");
                return false;
            }
        } catch (TimeoutException e) {
            future.cancel(true);
            debug("parsing tree timeout!");
            return false;
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        executor.shutdown();
        
        if(parsedTree.startsWith("(system_verilog_text (")) {
            return true;
        } else {
            debug("Does not start with system_verilog_text; this file cannot be parsed.");
            return false;
        }
    }

    private static boolean checkIfJavaParsable(CharStream charStream, int timeout) throws IOException {
        // We don't need to check if Java file is parsable. Always returns true.
        return true;
    }

    /* Check if the input hdl file can be parsed by antlr parser. If
     * the parsing operation takes 15 mins, the file is the file that
     * cannot be parsed.*/
    private static boolean checkIfParsable(String dataPath, String extension) throws RuntimeException {
        File inputFile = new File(dataPath);
        debug(inputFile.getName() + " ...");
        try {
            CharStream charStream = CharStreams.fromFileName(dataPath);
            int timeout = 15 * 60; // 15min

            if (extension.equals(World.VerilogExt)) {
                return checkIfVerilogParsable(charStream, timeout);
            } else if (extension.equals(World.VhdlExt)) {
                return checkIfVhdlParsable(charStream, timeout);
            } else if(extension.equals(World.SysVerExt)) {
                return checkIfSysVerParsable(charStream, timeout);
            } else if (extension.equals(World.JavaExt)) {
                return checkIfJavaParsable(charStream, timeout);
            } else {
                throw new RuntimeException("Input HDL file for cheking if parsable must be among Verilog, SystemVerilog, and VHDL");
            }
        } catch (Exception e) {
            debug(e.getClass());
            e.printStackTrace();
        }
        return false;
    }
    
    public static void main(String[] args) throws Exception {
        Options options = new Options();

        Option input = new Option("i", "repositorydir", true, "input repository directory");
        input.setRequired(true);
        options.addOption(input);

        Option output = new Option("e", "extension", true, "file extension");
        output.setRequired(true);
        options.addOption(output);
        
        Option dst = new Option("d", "dstDir", true, "destination directory");
        dst.setRequired(true);
        options.addOption(dst);

        Option lst = new Option("c", "fcOutput", true, "file that will keep results of checking");
        lst.setRequired(true);
        options.addOption(lst);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();

        FileWriter fcWriter = null;
        try {
            CommandLine cmd = parser.parse(options, args);
            String repoDir = cmd.getOptionValue("repositorydir");
            File repoFile = new File(repoDir);
            String fExt = cmd.getOptionValue("extension");
            String dstDir = cmd.getOptionValue("dstDir");
            String lstFileName = cmd.getOptionValue("fcOutput");
            fcWriter = new FileWriter(new File(lstFileName));

            debug(repoFile.getName() + " project: ");
            checkAllFiles(repoFile.getName(), repoDir, fExt, fcWriter);
        } catch (IOException ex) {
            ex.printStackTrace();
            System.exit(1);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("java FileChecker -i <repository directory> -e <file extension> -d <destination directory>", options);
            System.exit(1);
        } finally {
            FileUtil.closeWithoutException(fcWriter);
        }
        System.exit(0);
    }
}
