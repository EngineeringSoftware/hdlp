package slp.core.lexing.code;

import java.util.*;
import java.util.stream.Stream;

import slp.core.lexing.Lexer;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CodePointCharStream;

public class SystemVerilogSLPLexer implements Lexer {

	@Override
	public Stream<String> lexLine(String line) {
		int offset = line.indexOf("//");
		if (-1 != offset) {
			line = line.substring(0, offset);
		}
		CodePointCharStream charStream = CharStreams.fromString(line);
		SystemVerilogLexer lexer = new SystemVerilogLexer(charStream);
		List<String> tokens_list = new ArrayList<String>();
		while(true) {
			String token = lexer.nextToken().getText();
			if(token.equals("<EOF>")) {
				break;
			}
			tokens_list.add(token);
		}
		String[] tokens_arr = new String[tokens_list.size()];
		tokens_arr = tokens_list.toArray(tokens_arr);
		return Arrays.stream(tokens_arr);
	}
	
}
