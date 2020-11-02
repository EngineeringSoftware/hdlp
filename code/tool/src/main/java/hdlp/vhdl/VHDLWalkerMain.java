package hdlp.vhdl;

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
public class VHDLWalkerMain {
    
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

            // For obtaining assignments.json, concurrent signal assignment listener
            // and entity listener is required.
            File outputFile = outputDir.toPath().resolve(nameNoExt + ".asg").toFile();
            File outputTypeFile = outputDir.toPath().resolve(nameNoExt + ".typ").toFile();
            File outputEntityFile = outputDir.toPath().resolve(nameNoExt + ".ent").toFile();
            File outputFunctionFile = outputDir.toPath().resolve(nameNoExt + ".fn").toFile();
            File outputProcessFile = outputDir.toPath().resolve(nameNoExt + ".pcs").toFile();
            File outputIOPortFile = outputDir.toPath().resolve(nameNoExt + ".io").toFile();
            File outputAssignLocFile = outputDir.toPath().resolve(nameNoExt + ".asgloc").toFile();
            File outputAssertFile = outputDir.toPath().resolve(nameNoExt + ".ast").toFile();
            
            File outputStFile = outputDir.toPath().resolve(nameNoExt + ".st").toFile();
            
            FileWriter outputFileWriter = null;
            FileWriter outputTypeFileWriter = null;
            FileWriter outputEntityFileWriter = null;
            FileWriter outputFunctionFileWriter = null;
            FileWriter outputProcessFileWriter = null;
            FileWriter outputIOPortFileWriter = null;
            FileWriter outputAssignLocFileWriter = null;
            FileWriter outputAssertFileWriter = null;
            FileWriter outputStFileWriter = null;
            
            try {
                outputFileWriter = new FileWriter(outputFile);
                outputTypeFileWriter = new FileWriter(outputTypeFile);
                outputEntityFileWriter = new FileWriter(outputEntityFile);
                outputFunctionFileWriter = new FileWriter(outputFunctionFile);
                outputProcessFileWriter = new FileWriter(outputProcessFile);
                outputIOPortFileWriter = new FileWriter(outputIOPortFile);
                outputAssignLocFileWriter = new FileWriter(outputAssignLocFile);
                outputAssertFileWriter = new FileWriter(outputAssertFile);
                outputStFileWriter = new FileWriter(outputStFile);
                
                vhdlLexer lexer = new vhdlLexer(CharStreams.fromPath(inputFile.toPath()));
                CommonTokenStream tokens = new CommonTokenStream(lexer);
                vhdlParser parser = new vhdlParser(tokens);

                parser.addParseListener(new VHDLTypeListener(outputTypeFileWriter));
                parser.addParseListener(new VHDLEntityListener(outputFileWriter,
                                                               outputTypeFileWriter,
                                                               outputEntityFileWriter));
                
                parser.addParseListener(new VHDLFunctionListener(outputFunctionFileWriter));
                parser.addParseListener(new VHDLProcessListener(outputProcessFileWriter));
                parser.addParseListener(new VHDLIOPortListener(outputIOPortFileWriter));
                parser.addParseListener(new VHDLComponentListener(outputTypeFileWriter));
                parser.addParseListener(new VHDLAssignLOCListener(outputAssignLocFileWriter));
                parser.addParseListener(new VHDLAssertionListener(outputAssertFileWriter));
                parser.addParseListener(new VHDLStatementListener(outputStFileWriter));

                if (type.equalsIgnoreCase("seq")) {
                    parser.addParseListener(new VHDLContextAssignSeqListener(outputFileWriter));
                } else {
                    parser.addParseListener(new VHDLContextAssignParallelListener(outputFileWriter));
                }
                
                ParseTree tree = parser.design_file(); // parse
            } catch (IOException e) {
                throw new RuntimeException(e);
            } finally {
                FileUtil.closeWithoutException(outputFileWriter);
                FileUtil.closeWithoutException(outputTypeFileWriter);
                FileUtil.closeWithoutException(outputEntityFileWriter);
                FileUtil.closeWithoutException(outputFunctionFileWriter);
                FileUtil.closeWithoutException(outputProcessFileWriter);
                FileUtil.closeWithoutException(outputIOPortFileWriter);
                FileUtil.closeWithoutException(outputAssignLocFileWriter);
                FileUtil.closeWithoutException(outputAssertFileWriter);
                FileUtil.closeWithoutException(outputStFileWriter);
                FileUtil.deleteFileIfNoContent(outputFile);
                FileUtil.deleteFileIfNoContent(outputTypeFile);
                FileUtil.deleteFileIfNoContent(outputEntityFile);
                FileUtil.deleteFileIfNoContent(outputFunctionFile);
                FileUtil.deleteFileIfNoContent(outputProcessFile);
                FileUtil.deleteFileIfNoContent(outputIOPortFile);
                FileUtil.deleteFileIfNoContent(outputAssignLocFile);
                FileUtil.deleteFileIfNoContent(outputAssertFile);
                FileUtil.deleteFileIfNoContent(outputStFile);
            }
        }
    }
}
