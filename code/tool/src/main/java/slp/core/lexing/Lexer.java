package slp.core.lexing;

import java.io.File;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.antlr.v4.runtime.Token;
import slp.core.io.Reader;
import slp.core.lexing.JavaLexer;
import slp.core.lexing.code.Verilog2001CustomizedListener;
import slp.core.lexing.code.Verilog2001Lexer;
import slp.core.lexing.code.Verilog2001Parser;
import slp.core.lexing.code.vhdlLexer;
import slp.core.lexing.code.SystemVerilogLexer;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.CodePointCharStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

public interface Lexer {
	
	/**
	 * Lex all the lines in the provided file. Use of this method is preferred, since some Lexers benefit from knowing
	 * the file path (e.g. AST Lexers can use this for type inference).
	 * By default, invokes {@link #lexText(String)} with content of file.
	 * 
	 * @param file The file to be lexed
	 * @return A Stream of lines, where every line is lexed to a Stream of tokens
	 */
	default Stream<Stream<String>> lexFile(File file) {
		String fileName = file.getName();
		String ext = "";
		if(fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0) {
			ext = fileName.substring(fileName.lastIndexOf(".")+1);
		}
		else {
			System.err.println("File extension is not found.");
        }
		if (!ext.toLowerCase().equals("vhd") && !ext.toLowerCase().equals("v") && !ext.toLowerCase().equals("java") && !ext.toLowerCase().equals("sv")) {
			throw new RuntimeException("Unsupported language for file with extension " + ext);
		}

		// Lex the file and skip comments
		List<List<String>> lexed = new ArrayList<>();
		List<String> line = null;
		List<String> token_list = new ArrayList<String>();
		try {
			CharStream charStream = CharStreams.fromFileName(file.getAbsolutePath());
			// Init lexer, comment patterns according to language
			org.antlr.v4.runtime.Lexer lexer;
			String lineCommentPattern, blockCommentBeginPattern, blockCommentEndPattern;
			switch (ext.toLowerCase()) {
				case "vhd":
					lexer = new vhdlLexer(charStream);
					lineCommentPattern = "--";
					blockCommentBeginPattern = "/*";
					blockCommentEndPattern = "*/";
					break;
				case "java":
					lexer = new JavaLexer(charStream);
					lineCommentPattern = "//";
					blockCommentBeginPattern = "/*";
					blockCommentEndPattern = "*/";
					break;
				case "sv":
					lexer = new SystemVerilogLexer(charStream);
					lineCommentPattern = "//";
					blockCommentBeginPattern = "/*";
					blockCommentEndPattern = "*/";
					break;
				case "v":
					lexer = new Verilog2001Lexer(charStream);
					lineCommentPattern = "//";
					blockCommentBeginPattern = "/*";
					blockCommentEndPattern = "*/";
					break;
				default:
					throw new RuntimeException("Unknown language");
			}
			
			lexer.removeErrorListeners();
			
			int lastLineNo = -1;
			while(true) {
				Token token;
				String text;
				int lineNo = -1;
				
				token = lexer.nextToken();
				text = token.getText();
				lineNo = token.getLine();
				
				if (text.equals("<EOF>")) {
					break;
				}

				// Skip comments
				int lineCommentIndex = text.indexOf(lineCommentPattern);
				int blockCommentBegin = text.indexOf(blockCommentBeginPattern);
				int blockCommentEnd = text.indexOf(blockCommentEndPattern);
					
				if (-1!=blockCommentBegin && -1!=blockCommentEnd) {
					continue;
				} else if (-1!=lineCommentIndex) {
					continue;
				}
				
				// Skip empty text
				if (text.trim().isEmpty()) {
					continue;
				}
				
				// Put into appropriate line
				if (lineNo == lastLineNo) {
					if (line == null) {
						line = new ArrayList<>();
					}
					line.add(text);
				} else {
					if (line != null) {
						lexed.add(line);
						line = null;
					}
				}
				
				lastLineNo = lineNo;
			}
			
			if (line != null) {
				lexed.add(line);
				line = null;
			}
		} catch (Exception e) {
        	System.err.println(e.getClass());
            e.printStackTrace();
        }

		return lexed.stream().map(Collection::stream);
	}
	
	default Stream<Stream<String>> lexVerilogStatementFile(File file) {

		String fileName = file.getName();
		String ext = "";
		if(fileName.lastIndexOf(".") != -1 && fileName.lastIndexOf(".") != 0) {
			ext = fileName.substring(fileName.lastIndexOf(".")+1);
		}
		else {
			System.err.println("File extension is not found.");
			System.exit(0);
        }
		if (!ext.toLowerCase().equals("v")) {
			System.err.println("lexVerilogStatementFile method is only applied for verilog file (.v)");
			System.exit(0);
		}
		
		List<String> token_list = new ArrayList<String>();
		try {
			CharStream charStream = CharStreams.fromFileName(file.getAbsolutePath());
			Verilog2001Lexer lexer_veril = new Verilog2001Lexer(charStream);
			lexer_veril.removeErrorListeners();
			
			CommonTokenStream stream = new CommonTokenStream(lexer_veril);
			Verilog2001Parser verilparser = new Verilog2001Parser(stream);
			verilparser.removeErrorListeners();
			ParseTree verilptree = verilparser.source_text();
			String stringparsetree = verilptree.toStringTree(verilparser);
			if(stringparsetree.startsWith("(source_text (")) {
				ParseTreeWalker walker = new ParseTreeWalker();
				Verilog2001CustomizedListener listener = new Verilog2001CustomizedListener();
				walker.walk(listener, verilptree);
				token_list = listener.getstatementTokens();
			}
			else {
				System.out.println("Failed to parse.");
			}
		}
		catch (Exception e) {
        	System.err.println(e.getClass());
            e.printStackTrace();
        }
		String[] tokens = token_list.stream().filter(t -> !t.trim().isEmpty()).toArray(String[]::new);
		return Arrays.stream(tokens).map(t -> Stream.of(t));
	}

	/**
	 * Lex the provided text. The default implementation invokes {@linkplain #lexLine(String)} on each line in the text,
	 * but sub-classes may opt to lex the text as a whole instead (e.g. JavaLexer needs to do so to handle comments correctly).
	 * 
	 * @param file The text to be lexed
	 * @return A Stream of lines, where every line is lexed to a Stream of tokens
	 */

	default Stream<Stream<String>> lexText(String text) {
		return Arrays.stream(text.split("\n")).map(this::lexLine);
	}

	/**
	 * Lex the provided line into a stream of tokens.
	 * The default implementations of {@link #lexFile(File)} and {@link #lexText(String)} refer to this method,
	 * but sub-classes may override that behavior to take more advantage of the full content.
	 * 
	 * @param line The line to be lexed
	 * @return A Stream of tokens that are present on this line (may be an empty Stream).
	 */
	Stream<String> lexLine(String line);
}
