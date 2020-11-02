package slp.core.lexing.code;

import java.util.*;
import java.util.stream.Stream;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import slp.core.lexing.Lexer;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CodePointCharStream;

public class verilogLexer implements Lexer {

	@Override
	public Stream<String> lexLine(String line) {
		int lcomm_indx = line.indexOf("//");
		if (-1 != lcomm_indx) {
			line = line.substring(0, lcomm_indx);
		}
		
		int pcomm_from_indx = line.indexOf("/*");
		int pcomm_to_indx = line.indexOf("*/");
		if (-1!=pcomm_from_indx && -1!=pcomm_to_indx) {			
			String line_from = line.substring(0, pcomm_from_indx);
			String line_to = line.substring(pcomm_to_indx+1);
			line = String.join(line_from, line_to);
		}
		else if (-1!=pcomm_from_indx && -1==pcomm_to_indx) {
			line = line.substring(0, pcomm_from_indx);
		}
		else if (-1==pcomm_from_indx && -1!=pcomm_to_indx) {
			line = line.substring(pcomm_to_indx);
		}
		
		CodePointCharStream charStream = CharStreams.fromString(line);
		Verilog2001Lexer lexer = new Verilog2001Lexer(charStream);
		List<String> tokens_list = new ArrayList<String>();
		while(true) {
			String token = lexer.nextToken().getText();
			if(token.equals("<EOF>")) {
				break;
			}
			tokens_list.add(token);
		}
		tokens_list = tokens_list.stream().filter(t -> !t.trim().isEmpty()).collect(Collectors.toList());
		String[] tokens_arr = new String[tokens_list.size()];
		tokens_arr = tokens_list.toArray(tokens_arr);
		return Arrays.stream(tokens_arr);
	}
	
}
