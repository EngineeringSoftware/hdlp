readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export JAVA6_HOME=$JAVA_HOME
export JAVA7_HOME=$JAVA_HOME
export JAVA8_HOME=$JAVA_HOME



readonly TOOL_DIR=$( cd "${_DIR}/../../../../../tool" && pwd )
readonly BIN_DIR="${TOOL_DIR}/build/classes/java/main"
readonly LIB_DIR="${TOOL_DIR}/libs"

CLASSPATH=${BIN_DIR}
for j in $(find "$(cd ${LIB_DIR}; pwd)" -name "*.jar" | sort); do
        CLASSPATH=${j}:$CLASSPATH
done

CLASSPATH=${_DIR}:$CLASSPATH

# Tools.
readonly JAVACCMD_PARSER="javac -classpath ${CLASSPATH} -d ${BIN_DIR} ./vhdl/vhdlParser.java"
readonly JAVACCMD_BLISTENER="javac -classpath ${CLASSPATH} -d ${BIN_DIR} ./vhdl/vhdlBaseListener.java"
readonly JAVACCMD_LISTENER="javac -classpath ${CLASSPATH} -d ${BIN_DIR} ./vhdl/VHDLContextAssignListener.java"
readonly JAVACCMD="javac -classpath ${CLASSPATH} -d ${BIN_DIR} example.java"
readonly JAVACMD="java -classpath ${CLASSPATH} hdlp.example -i sha256_qp.vhd"

${JAVACCMD_PARSER}
${JAVACCMD_BLISTENER}
${JAVACCMD_LISTENER}
${JAVACCMD}
${JAVACMD} 
