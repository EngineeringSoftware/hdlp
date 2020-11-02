package hdlp.util;

import java.io.Writer;
import java.io.IOException;
import java.io.File;

/**
 * Util methods for dealing with IO.
 */
public class FileUtil {

    public static void closeWithoutException(Writer w) {
        try {
            if (w != null) {
                w.flush();
                w.close();
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static void deleteFileIfNoContent(String fileName) {
        deleteFileIfNoContent(new File(fileName));
    }
    
    public static void deleteFileIfNoContent(File f) {
        if (f.length() == 0) {
            f.delete();
        }
    }
}
