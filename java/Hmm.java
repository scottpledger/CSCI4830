/** This is a template for an HMM class.  Fill in code for the
 * constructor and all of the methods.  Do not change the signature of
 * any of these, and do not add any other public fields, methods or
 * constructors (but of course it is okay to add private stuff).  All
 * public access to this class must be via the constructor and methods
 * specified here.
 */
public class Hmm {

    /** Constructs an HMM from the given data.  The HMM will have
     * <tt>numStates</tt> possible states and <tt>numOutputs</tt>
     * possible outputs.  The HMM is then built from the given set of
     * state and output sequences.  In particular,
     * <tt>state[i][j]</tt> is the <tt>j</tt>-th element of the
     * <tt>i</tt>-th state sequence, and similarly for
     * <tt>output[i][j]</tt>.
     */
    public Hmm(int numStates, int numOutputs,
	       int state[][], int output[][]) {

	// your code here

    }

    /** Returns the number of states in this HMM. */
    public int getNumStates() {

	// your code here

    }

    /** Returns the number of output symbols for this HMM. */
    public int getNumOutputs() {

	// your code here

    }

    /** Returns the log probability assigned by this HMM to a
     * transition from the dummy start state to the given
     * <tt>state</tt>.
     */
    public double getLogStartProb(int state) {

	// your code here

    }

    /** Returns the log probability assigned by this HMM to a
     * transition from <tt>fromState</tt> to <tt>toState</tt>.
     */
    public double getLogTransProb(int fromState, int toState) {

	// your code here

    }

    /** Returns the log probability of <tt>state</tt> emitting
     * <tt>output</tt>.
     */
    public double getLogOutputProb(int state, int output) {

	// your code here

    }

}
