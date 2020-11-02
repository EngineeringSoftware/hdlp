package hdlp.util;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.ParseException;

import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.FileAlreadyExistsException;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.Scanner;
import java.util.Arrays;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.List;

public class AssignmentJSON {

    private static List<File> findFilesRec(File inDir, String ext) {
        List<File> result = new ArrayList<File>();
        for (File f : inDir.listFiles()) {
            if (f.isDirectory()) {
                result.addAll(findFilesRec(f, ext));
            } else if (f.getAbsolutePath().endsWith(ext)) {
                result.add(f);
            }
        }
        return result;
    }

    private static List<List<String>> splitEntityStrings(File extractedFile) {
        List<List<String>> resultList = new ArrayList<List<String>>();
        ArrayList<String> linesForEntity = new ArrayList<String>();
        int indx = 0;
        boolean isComp = false;
        try {
            Scanner fileReader = new Scanner(extractedFile);
            while (fileReader.hasNextLine()){
                String curLine = fileReader.nextLine().toLowerCase();
                if(curLine.contains("<entity>")) {
                    if (indx>0 && linesForEntity.size()>0) {
                        ArrayList<String> temp = new ArrayList<String>();
                        temp.addAll(linesForEntity);
                        resultList.add(temp);
                        linesForEntity.clear();
                    }
                    linesForEntity.add(curLine.trim());
                } else if (curLine.contains("<component>")) {
                    isComp = true;
                    
                } else if (curLine.contains("</component>")) {
                    isComp = false;
                    
                } else {
                    if (!isComp) { linesForEntity.add(curLine.trim()); }
                }
                indx++;
            }
            resultList.add(linesForEntity);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return resultList;
    }
        
    private static void createJson(File inDir, Path outDir, String type) throws IOException {
        File[] files = findFilesRec(inDir, ".asg").toArray(new File[0]);
        Arrays.sort(files, new Comparator<File>() {
                @Override
                public int compare(File f1, File f2) {
                    return f1.getAbsolutePath().compareTo(f2.getAbsolutePath());
                }
            });

        PrintWriter pw = null;
        try {
            pw = new PrintWriter(Files.createFile(outDir).toFile());
        } catch (FileAlreadyExistsException ex){
            System.out.println("Json File already exists. overwritting");
            pw = new PrintWriter(outDir.toFile());
        }

        pw.println("[");

        for(int i = 0; i < files.length; i++) {
            File curFile = files[i];
            String curTypeFilePath = curFile.getAbsolutePath().replace(".asg", ".typ");
            File curTypeFile = new File(curTypeFilePath);
            List<List<String>> agnLineList = splitEntityStrings(curFile);
            List<List<String>> typLineList = null;
            if (curTypeFile.exists()) { typLineList = splitEntityStrings(curTypeFile); }
            
            pw.println("{");

            pw.println("\"fn\" : \"" + inDir.toPath().relativize(curFile.toPath()).toString() + ", " + inDir.toPath().relativize(curTypeFile.toPath()).toString() + "\",");
            pw.println("\"entity\" : [ {");
            for (int j = 0; j < agnLineList.size(); j++) {
                if (curTypeFile.exists()) {
                    int typ_j = -1;
                    for (int k = 0; k < typLineList.size(); k++) {
                        if (typLineList.get(k).get(0).equals(agnLineList.get(j).get(0))) {
                            typ_j = k;
                        }
                    }
                    if (agnLineList.get(j).get(0).contains("<entity>")) {
                        pw.println("\"name\" : \"" + agnLineList.get(j).get(0) + "\",");
                    } else {
                        pw.println("\"name\" : \"\",");
                    }
                    if (typ_j>-1) {
                        pw.print("\"type\" : { ");
                        for (int k = 0; k < typLineList.get(typ_j).size(); k++) {
                            String curLine = typLineList.get(typ_j).get(k).toLowerCase();

                            if (curLine.length() == 0) {
                                continue;
                            }
                            String curString = curLine.trim();
                    
                            // replace "is" with ":"
                            int isIndex = curString.lastIndexOf(" is ");
                            if (isIndex>-1) {
                                curString = curString.replace(" is ", " : ");
                            }

                            // remove initalization tokens
                            int initValIndex = curString.lastIndexOf(":=");
                            if (initValIndex>-1) {
                                curString = curString.substring(0, initValIndex);
                            }
                    
                            String[] varTypeTokens = curString.trim().split(":");
                            if (varTypeTokens.length>1) {
                                String[] keys = varTypeTokens[0].split(",");
                                String val = "\"" + varTypeTokens[1].trim() + "\"";
                                for(int l = 0; l <keys.length; l++) {
                                    String key = "\"" + keys[l].trim() + "\"";
                                    pw.print(key + " : " + val);
                                    if (l<keys.length-1) {
                                        pw.print(", ");
                                    }
                                }
                                if (k<typLineList.get(typ_j).size()-1) {
                                    pw.print(", ");
                                }
                            }
                        }
                        pw.println(" },");
                    } else {
                        pw.print("\"type\" : { },");
                    }
                } else {
                    pw.println("\"name\" : \"\",");
                    pw.print("\"type\" : { },");
                }
                pw.print("\"agn\" : [ ");
                for (int p = 1; p < agnLineList.get(j).size(); p++) {
                    String curLine = agnLineList.get(j).get(p).toLowerCase();
                    if (curLine.length() == 0) {
                        continue;
                    }
                    pw.println("{");
                    String curString = curLine.trim();
                    int prevAssignToIndex = curString.lastIndexOf("<prevassign>");
                    String prevAssignString = curString.substring(0, prevAssignToIndex);
                    pw.print("\"prevassign\" : [ [");
                    String[] prevAssignTokens = prevAssignString.trim().split(" ");
                    for (int q = 0; q < prevAssignTokens.length; q++){
                        if (prevAssignTokens[q].equals("<prevassign>")) {
                            pw.println("], [");
                        } else {
                            pw.print("\"" + prevAssignTokens[q].replace("\"", "\\\"") + "\"");
                            if (q != prevAssignTokens.length - 1 && !prevAssignTokens[q+1].equals("<prevassign>")) {
                                pw.print(", ");
                            }
                        }
                    }
                    pw.println("] ],");
                    
                    int contextToIndex = curString.lastIndexOf("<context>");
                    String contextString = curString.substring(prevAssignToIndex+12, contextToIndex);
                    pw.print("\"pcontext\" : [ ");
                    String[] contextTokens = contextString.trim().split(" ");
                    for (int q = 0; q < contextTokens.length; q++){
                        pw.print("\"" + contextTokens[q].replace("\"", "\\\"") + "\"");
                        if (q != contextTokens.length-1) {
                            pw.print(", ");
                        }
                    }
                    pw.println("],");

                    String contextlhsString = curString.substring(contextToIndex+9);
                    int delimeterIndex = contextlhsString.indexOf("<=");
                    
                    pw.print("\"l\" : [ ");
                    String lhsString = contextlhsString.substring(0, delimeterIndex);
                    String[] lhsTokens = lhsString.trim().split(" ");
                    for (int q = 0; q < lhsTokens.length; q++){
                        pw.print("\"" + lhsTokens[q].replace("\"", "\\\"") + "\"");
                        if (q != lhsTokens.length-1) {
                            pw.print(", ");
                        }
                    }
                    pw.println("],");
                    
                    pw.print("\"r\" : [ \"<=\", ");
                    String rhsString = contextlhsString.substring(delimeterIndex+2);
                    String[] rhsTokens = rhsString.trim().split(" ");
                    for (int q = 0; q < rhsTokens.length; q++){                    
                        pw.print("\"" + rhsTokens[q].replace("\"", "\\\"") + "\"");
                        if (q != rhsTokens.length-1) {
                            pw.print(", ");
                        }
                    }
                    pw.println("]");
                    if (p < agnLineList.get(j).size()-1) {
                        pw.println("},");
                    } else {
                        pw.println("}");
                    }
                }
                pw.println("]");
                if (j < agnLineList.size()-1) {
                    pw.println("}, {");
                } else {
                    pw.println("} ]");
                }
            }
            if (i != files.length-1) {
                pw.println("}, ");
            } else {
                pw.print("}");
            }
        }

        pw.println("]");
        pw.flush();
        pw.close();
    }
    
    public static void main(String[] args){
        Options options = new Options();

        Option optInput = new Option("i", "inputAsg", true, "the path to the directory containing .asg files");
        optInput.setRequired(true);
        options.addOption(optInput);

        Option optOutput = new Option("o", "output", true, "the path to the output .json file");
        optOutput.setRequired(true);
        options.addOption(optOutput);
    
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        try {
            CommandLine cmd = parser.parse(options, args);
            String type = cmd.getOptionValue("type");
            Path outPath = Paths.get(cmd.getOptionValue("output"), "assignments.json");
            File asgDir = new File(cmd.getOptionValue("inputAsg"));
            createJson(asgDir, outPath, type);
        } catch (IOException ex) {
            ex.printStackTrace();
            System.exit(1);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("java AssignmentJSON -i input_dir -o output_dir -t type", options);
            System.exit(1);
        }
    }
}
