package hdlp.sysverilog;

import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.CharStreams;

import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.List;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Objects;

import hdlp.util.FileUtil;

/**
 * Util class for finding functions and concurrent assignments.
 */
public class SystemVerilogWalkerMain {
    
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
            
            File outputFile = outputDir.toPath().resolve(nameNoExt + ".asg").toFile();
            File outputModuleFile = outputDir.toPath().resolve(nameNoExt + ".mod").toFile();
            File outputFunctionFile = outputDir.toPath().resolve(nameNoExt + ".fn").toFile();
            File outputAlwaysFile = outputDir.toPath().resolve(nameNoExt + ".alw").toFile();
            File outputInoutPortFile = outputDir.toPath().resolve(nameNoExt + ".iop").toFile();
            File outputInputPortFile = outputDir.toPath().resolve(nameNoExt + ".inp").toFile();
            File outputOutputPortFile = outputDir.toPath().resolve(nameNoExt + ".outp").toFile();
            File outputAssertPropertyFile = outputDir.toPath().resolve(nameNoExt + ".ast").toFile();
            
            FileWriter outputFileWriter = null;
            FileWriter outputModuleFileWriter = null;
            FileWriter outputFunctionFileWriter = null;
            FileWriter outputAlwaysFileWriter = null;
            FileWriter outputInoutPortFileWriter = null;
            FileWriter outputInputPortFileWriter = null;
            FileWriter outputOutputPortFileWriter = null;
            FileWriter outputAssertPropertyFileWriter = null;
            try {
                outputFileWriter = new FileWriter(outputFile);
                outputModuleFileWriter = new FileWriter(outputModuleFile);
                outputFunctionFileWriter = new FileWriter(outputFunctionFile);
                outputAlwaysFileWriter = new FileWriter(outputAlwaysFile);
                outputInoutPortFileWriter = new FileWriter(outputInoutPortFile);
                outputInputPortFileWriter = new FileWriter(outputInputPortFile);
                outputOutputPortFileWriter = new FileWriter(outputOutputPortFile);
                outputAssertPropertyFileWriter = new FileWriter(outputAssertPropertyFile);

                SystemVerilogLexer lexer = new SystemVerilogLexer(CharStreams.fromPath(inputFile.toPath()));
                CommonTokenStream tokens = new CommonTokenStream(lexer);
                SystemVerilogParser parser = new SystemVerilogParser(tokens);

                parser.addParseListener(new SystemVerilogModuleListener(outputModuleFileWriter));
                parser.addParseListener(new SystemVerilogFunctionListener(outputFunctionFileWriter));
                parser.addParseListener(new SystemVerilogAlwaysListener(outputAlwaysFileWriter));
                parser.addParseListener(new SystemVerilogInoutPortListener(outputInoutPortFileWriter));
                parser.addParseListener(new SystemVerilogInputPortListener(outputInputPortFileWriter));
                parser.addParseListener(new SystemVerilogOutputPortListener(outputOutputPortFileWriter));
                parser.addParseListener(new SystemVerilogAssertPropertyListener(outputAssertPropertyFileWriter));
         
                ParseTree tree = parser.system_verilog_text(); // parse
            } catch (IOException e) {
                throw new RuntimeException(e);
            } finally {
                FileUtil.closeWithoutException(outputFileWriter);
                FileUtil.closeWithoutException(outputModuleFileWriter);
                FileUtil.closeWithoutException(outputFunctionFileWriter);
                FileUtil.closeWithoutException(outputAlwaysFileWriter);
                FileUtil.closeWithoutException(outputInoutPortFileWriter);
                FileUtil.closeWithoutException(outputInputPortFileWriter);
                FileUtil.closeWithoutException(outputOutputPortFileWriter);
                FileUtil.closeWithoutException(outputAssertPropertyFileWriter);
                FileUtil.deleteFileIfNoContent(outputFile);
                FileUtil.deleteFileIfNoContent(outputModuleFile);
                FileUtil.deleteFileIfNoContent(outputFunctionFile);
                FileUtil.deleteFileIfNoContent(outputAlwaysFile);
                FileUtil.deleteFileIfNoContent(outputInoutPortFile);
                FileUtil.deleteFileIfNoContent(outputInputPortFile);
                FileUtil.deleteFileIfNoContent(outputOutputPortFile);
                FileUtil.deleteFileIfNoContent(outputAssertPropertyFile);
            }
        }
        
    }
}
