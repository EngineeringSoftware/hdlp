package hdlp.verilog;

import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CharStream;

import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.io.Writer;
import java.io.FileWriter;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Objects;

import hdlp.util.FileUtil;

/**
 * Util class for finding functions and concurrent assignments.
 */
public class Verilog2001WalkerMain {
    
    /**
     * Finds all concurrent assignments (according to the given type) from the input files, extract to output files.
     */
    public static void main(String[] args) {
        
        // Check inputs
        if (args.length != 4) {
            throw new RuntimeException("Wrong number of arguments! Expect: input_dir, type, blacklist_file, output_dir");
        }
        
        File inputDir = new File(args[0]);
        String type = args[1];
        File blackListFile = new File(args[2]);
        File outputDir = new File(args[3]);
        System.out.println(blackListFile.getAbsolutePath());

        List<String> blackListFileNames;
        try {
            
            blackListFileNames = Files.readAllLines(blackListFile.toPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    
        // The order does not matter here -- we generate the output files with the same names of input files
        for (File inputFile: Objects.requireNonNull(inputDir.listFiles())) {
            String nameNoExt = FilenameUtils.removeExtension(inputFile.getName());
            // Skip files in blacklists
            if (blackListFileNames.contains(nameNoExt)) {
                continue;
            }
            
            File outputModuleFile = outputDir.toPath().resolve(nameNoExt + ".mod").toFile();
            File outputFunctionFile = outputDir.toPath().resolve(nameNoExt + ".fn").toFile();
            File outputAlwaysFile = outputDir.toPath().resolve(nameNoExt + ".alw").toFile();
            File outputInoutPortFile = outputDir.toPath().resolve(nameNoExt + ".iop").toFile();
            File outputInputPortFile = outputDir.toPath().resolve(nameNoExt + ".inp").toFile();
            File outputOutputPortFile = outputDir.toPath().resolve(nameNoExt + ".outp").toFile();
            
            FileWriter outputModuleFileWriter = null;
            FileWriter outputFunctionFileWriter = null;
            FileWriter outputAlwaysFileWriter = null;
            FileWriter outputInoutPortFileWriter = null;
            FileWriter outputInputPortFileWriter = null;
            FileWriter outputOutputPortFileWriter = null;
            try {
                outputModuleFileWriter = new FileWriter(outputModuleFile);
                outputFunctionFileWriter = new FileWriter(outputFunctionFile);
                outputAlwaysFileWriter = new FileWriter(outputAlwaysFile);
                outputInoutPortFileWriter = new FileWriter(outputInoutPortFile);
                outputInputPortFileWriter = new FileWriter(outputInputPortFile);
                outputOutputPortFileWriter = new FileWriter(outputOutputPortFile);

                Verilog2001Lexer lexer = new Verilog2001Lexer(CharStreams.fromPath(inputFile.toPath()));
                CommonTokenStream tokens = new CommonTokenStream(lexer);
                Verilog2001Parser parser = new Verilog2001Parser(tokens);

                parser.addParseListener(new Verilog2001ModuleListener(outputModuleFileWriter));
                parser.addParseListener(new Verilog2001FunctionListener(outputFunctionFileWriter));
                parser.addParseListener(new Verilog2001AlwaysListener(outputAlwaysFileWriter));
                parser.addParseListener(new Verilog2001InoutPortListener(outputInoutPortFileWriter));
                parser.addParseListener(new Verilog2001InputPortListener(outputInputPortFileWriter));
                parser.addParseListener(new Verilog2001OutputPortListener(outputOutputPortFileWriter));
         
                ParseTree tree = parser.source_text(); // parse
            } catch (IOException e) {
                throw new RuntimeException(e);
            } finally {
                FileUtil.closeWithoutException(outputModuleFileWriter);
                FileUtil.closeWithoutException(outputFunctionFileWriter);
                FileUtil.closeWithoutException(outputAlwaysFileWriter);
                FileUtil.closeWithoutException(outputInoutPortFileWriter);
                FileUtil.closeWithoutException(outputInputPortFileWriter);
                FileUtil.closeWithoutException(outputOutputPortFileWriter);
                FileUtil.deleteFileIfNoContent(outputModuleFile);
                FileUtil.deleteFileIfNoContent(outputFunctionFile);
                FileUtil.deleteFileIfNoContent(outputAlwaysFile);
                FileUtil.deleteFileIfNoContent(outputInoutPortFile);
                FileUtil.deleteFileIfNoContent(outputInputPortFile);
                FileUtil.deleteFileIfNoContent(outputOutputPortFile);
            }
        }
        
    }
}
