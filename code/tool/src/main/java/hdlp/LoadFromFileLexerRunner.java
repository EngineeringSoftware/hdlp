package hdlp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import slp.core.lexing.Lexer;
import slp.core.lexing.runners.LexerRunner;
import slp.core.translating.Vocabulary;
import slp.core.util.Pair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class LoadFromFileLexerRunner extends LexerRunner {
    
    private static GsonBuilder GSON_BUILDER = new GsonBuilder()
            .disableHtmlEscaping();
    
    private File dataPath;
    public List<Integer> trainIndexes;
    public List<Integer> testIndexes;
    private boolean isTraining = true;
    
    public boolean isTraining() {
        return isTraining;
    }
    
    public void setTraining(boolean training) {
        isTraining = training;
    }
    
    public LoadFromFileLexerRunner(LexerRunner lexerRunner, File dataPath, List<Integer> trainIndexes, List<Integer> testIndexes) {
        super(lexerRunner.getLexer(), lexerRunner.isPerLine(), lexerRunner.isStatement());
        this.dataPath = dataPath;
        this.trainIndexes = trainIndexes;
        this.testIndexes = testIndexes;
    }
    
    // Cache of the lexed data
    private List<List<String>> lexedCache = null;
    
    private List<List<String>> getLexed() {
        if (lexedCache == null) {
            try {
                Gson gson = GSON_BUILDER.create();
                JsonReader jsonReader = gson.newJsonReader(new FileReader(dataPath));
    
                lexedCache = new ArrayList<>();
    
                jsonReader.beginArray();
                while (jsonReader.hasNext()) {
                    List<String> line = new ArrayList<>();
                    jsonReader.beginArray();
                    while (jsonReader.hasNext()) {
                        line.add(jsonReader.nextString());
                    }
                    jsonReader.endArray();
                    lexedCache.add(line);
                }
                jsonReader.endArray();
    
                System.out.println("Loaded lexed data of size " + lexedCache.size());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return lexedCache;
    }
    
    /**
     * Rather than lexing the files in parameter {@code directory}, load the data from the {@link #dataPath}, and take the data at {@link #trainIndexes} or {@link #testIndexes}.
     *
     * @param directory  Ignored
     * @return a singleton list of pairs of (file, lines lexed from the file), where file is null (as we load from data instead of original files).
     */
    @Override
    public Stream<Pair<File, Stream<Stream<String>>>> lexDirectory(File directory) {
        // Current approach load all files into a list ...
        List<List<String>> lexed = getLexed();
        
        // Output the stream of training/testing data
        List<Integer> indexes = isTraining() ? trainIndexes : testIndexes;
        return Stream.of(Pair.of(null,indexes.stream().map(i -> lexed.get(i).stream())));
    }
    
    @Override
    public Stream<Stream<String>> lexFile(File file) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public Stream<Stream<String>> lexText(String content) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public Stream<String> lexLine(String line) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void lexDirectory(File from, File to) {
        throw new UnsupportedOperationException();
    }
    
    @Override
    public void lexDirectoryToIndices(File from, File to, Vocabulary vocabulary) {
        throw new UnsupportedOperationException();
    }
}
