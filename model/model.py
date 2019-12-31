from utils.op import *
import tensorflow as tf
import numpy as np
from config.params import *
from config.constant import *


class ACNet(object):
    def __init__(self, scope,session,device,N_S,N_A,type,globalAC=None):
        tf.set_random_seed(50)

        with tf.device(device):
            self.dim_a = N_A
            self.dim_s = N_S
            self.session = session


            if scope == 'Global_Net':
                    with tf.variable_scope(scope):
                        self.s = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
                        self.t = tf.placeholder(tf.float32, [None, self.dim_s], 'Target')

                        self.global_a_params = self._build_global_params_dict(scope)
                        self.special_a_params_dict,self.special_c_params_dict = self._build_special_params_dict(scope)


            elif type == 'Target_Special':
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.s = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
                    self.a = tf.placeholder(tf.int32, [None, ], 'Action')
                    self.special_v_target = tf.placeholder(tf.float32, [None, 1], 'special_V_target')

                    self.OPT_SPE_A = tf.train.RMSPropOptimizer(LR_SPE_A, name='Spe_RMSPropA')
                    self.OPT_SPE_C = tf.train.RMSPropOptimizer(LR_SPE_C, name='Spe_RMSPropC')

                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params \
                        = self._build_special_net(scope)

                    self._prepare_special_loss(scope)

                    self._prepare_special_grads(scope)

                    self._prepare_special_update_op(scope)

                    self._prepare_special_pull_op(scope)

            elif type == 'Target_General':

                self.global_AC = globalAC

                self.s = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
                self.a = tf.placeholder(tf.int32, [None, ], 'Action')

                self.t = tf.placeholder(tf.float32, [None, self.dim_s], 'T')

                self.adv      = tf.placeholder(tf.float32, [None,1], 'Advantage')
                # self.kl_beta  = tf.placeholder(tf.float32, [None,], 'KL_BETA')

                self.OPT_A = tf.train.RMSPropOptimizer(LR_A, name='Glo_RMSPropA')

                # target_general_network
                self.global_a_prob ,self.global_a_params = self._build_global_net(scope)

                # target_special_network
                self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params \
                    = self._build_special_net(scope)

                self._prepare_global_loss(scope)

                self._prepare_global_grads(scope)

                self._prepare_global_update_op(scope)

                self._prepare_global_pull_op(scope)

                self._prepare_special_pull_op(scope)

                self._prepare_many_goal_loss_and_update()

    def _build_global_params_dict(self, scope):
        with tf.variable_scope(scope):
            # encode
            w_encode = generate_fc_weight(shape=[self.dim_s, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512]        , name='global_b_encode')
            # fusion
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512]        , name='global_b_f')
            # scene
            w_scene  = generate_fc_weight(shape=[512, 512]  , name='global_w_s')
            b_scene  = generate_fc_bias(shape=[512]         , name='global_b_s')
            # actor
            w_actor  = generate_fc_weight(shape=[512, self.dim_a] , name='global_w_a')
            b_actor  = generate_fc_bias(shape=[self.dim_a]        , name='global_b_a')

            a_params = [w_encode, b_encode, w_fusion, b_fusion,w_scene, b_scene, w_actor,  b_actor]

            return a_params

    def _build_special_params_dict(self,scope):
        with tf.variable_scope(scope):
            a_params_dict,c_params_dict = dict(),dict()
            for target_key in TARGET_ID_LIST:
                w_actor  = generate_fc_weight(shape=[self.dim_s, self.dim_a], name='actor_w'+str(target_key))
                b_actor  = generate_fc_bias(shape=[self.dim_a],               name='actor_b'+str(target_key))
                w_critic = generate_fc_weight(shape=[self.dim_s, 1],          name='critic_w'+str(target_key))
                b_critic = generate_fc_bias(shape=[1],                        name='critic_b'+str(target_key))
                a_params = [w_actor  ,b_actor ]
                c_params = [w_critic ,b_critic]
                kv_a = {target_key:a_params}
                kv_c = {target_key:c_params}
                a_params_dict.update(kv_a)
                c_params_dict.update(kv_c)
            return  a_params_dict,c_params_dict

    def _build_global_net(self, scope):
        with tf.variable_scope(scope):
            # global_network only need actor

            w_encode = generate_fc_weight(shape=[self.dim_s, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512], name='global_b_encode')

            # encode current_state into s_encode
            s_encode = tf.nn.elu(tf.matmul(self.s, w_encode) + b_encode)
            # encode target_state  into t_encode
            t_encode = tf.nn.elu(tf.matmul(self.t, w_encode) + b_encode) # encode target_state  into t_encode

            # s_encode||t_encode --> concat
            concat = tf.concat([s_encode, t_encode], axis=1)  # s_encode||t_encode --> concat

            # concat --> fusion_layer
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512], name='global_b_f')
            fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

            # fusion_layer --> scene_layer
            w_scene = generate_fc_weight(shape=[512, 512], name='global_w_s')
            b_scene = generate_fc_bias(shape=[512], name='global_b_s')
            scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

            # scene_layer --> prob
            w_actor = generate_fc_weight(shape=[512, self.dim_a], name='global_w_a')
            b_actor = generate_fc_bias(shape=[self.dim_a], name='global_b_a')
            self.global_logits = tf.matmul(scene_layer, w_actor) + b_actor
            prob = tf.nn.softmax(self.global_logits)

            a_params = [w_encode, b_encode,w_fusion, b_fusion,
                        w_scene , b_scene ,w_actor , b_actor ]

            return prob , a_params

    def _build_special_net(self, scope):
        with tf.variable_scope(scope):

            # special_actor
            w_actor = generate_fc_weight(shape=[self.dim_s, self.dim_a], name='special_w_a')
            b_actor = generate_fc_bias(shape=[self.dim_a], name='special_b_a')
            self.special_logits = tf.matmul(self.s, w_actor) + b_actor
            prob = tf.nn.softmax(self.special_logits)

            # special_critic
            w_critic = generate_fc_weight(shape=[self.dim_s, 1], name='special_w_c')
            b_critic = generate_fc_bias(shape=[1], name='special_b_c')
            value = tf.matmul(self.s, w_critic) + b_critic

            a_params = [w_actor,  b_actor ]
            c_params = [w_critic, b_critic]

            return prob, value, a_params, c_params

    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):

            with tf.name_scope('global_a_loss'):
                # prob from target_special_net (dont update)
                p_target = tf.stop_gradient(self.special_a_prob)
                # prob from global_special_net (need update)
                p_update = self.global_a_prob

                self.spe_actor_reg_loss = -tf.reduce_mean(p_target*tf.log(tf.clip_by_value(p_update,1e-10,1.0)))

                glo_log_prob = tf.reduce_sum(tf.log(self.global_a_prob + 1e-5)*tf.one_hot(self.a, self.dim_a, dtype=tf.float32),
                                             axis=1,keep_dims=True)

                # self.adv is calculated by target_special_network
                actor_loss = glo_log_prob *self.adv

                self.glo_entropy = -tf.reduce_mean(self.global_a_prob*tf.log(self.global_a_prob + 1e-5), axis=1,keep_dims=True)  # encourage exploration

                self.loss = ENTROPY_BETA*self.glo_entropy + actor_loss

                self.global_a_loss = tf.reduce_mean(-self.loss )

    def _prepare_special_loss(self,scope):
        with tf.name_scope(scope+'special_loss'):

            with tf.name_scope('special_c_loss'):
                self.special_td = tf.subtract(self.special_v_target, self.special_v, name='special_TD_error')
                self.special_c_loss = tf.reduce_mean(tf.square(self.special_td))

            with tf.name_scope('special_a_loss'):
                special_log_prob = tf.reduce_sum(
                    tf.log(self.special_a_prob+1e-9)*tf.one_hot(self.a,4,dtype=tf.float32),axis=1,keep_dims=True)
                spe_exp_v = special_log_prob*tf.stop_gradient(self.special_td)
                self.spe_entropy = -tf.reduce_mean(self.special_a_prob*tf.log(self.special_a_prob+1e-5),axis=1,keep_dims=True)  # encourage exploration
                self.spe_exp_v = ENTROPY_BETA*self.spe_entropy + spe_exp_v
                self.special_a_loss = tf.reduce_mean(-self.spe_exp_v)

    def _prepare_global_grads(self,scope):
        with tf.name_scope(scope+'global_grads'):
            with tf.name_scope('global_net_grad'):
                self.global_a_grads = [tf.clip_by_norm(item, 40) for item in
                                       tf.gradients(self.global_a_loss, self.global_a_params)]

    def _prepare_special_grads(self,scope):
        with tf.name_scope(scope+'special_grads'):
            with tf.name_scope('special_net_grad'):
                self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.special_a_loss, self.special_a_params)]

                self.special_c_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.special_c_loss, self.special_c_params)]

    def _prepare_global_update_op(self,scope):
        with tf.name_scope(scope+'_global_update'):
            self.update_global_a_op = self.OPT_A.apply_gradients(list(zip(self.global_a_grads, self.global_AC.global_a_params)))

    def _prepare_special_update_op(self,scope):
        with tf.name_scope(scope+'_special_update'):
            self.update_special_a_dict, self.update_special_c_dict = dict(), dict()
            self.update_special_q_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: self.OPT_SPE_A.apply_gradients(list(zip(self.special_a_grads , self.global_AC.special_a_params_dict[key])))}
                kv_c = {key: self.OPT_SPE_C.apply_gradients(list(zip(self.special_c_grads , self.global_AC.special_c_params_dict[key])))}
                self.update_special_a_dict.update(kv_a)
                self.update_special_c_dict.update(kv_c)

    def _prepare_global_pull_op(self,scope):
        with tf.name_scope(scope+'pull_global_params'):
            self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in
                                         zip(self.global_a_params, self.global_AC.global_a_params)]

    def _prepare_special_pull_op(self,scope):
        with tf.name_scope(scope+'pull_special_params'):
            self.pull_a_params_special_dict, self.pull_c_params_special_dict = dict(), dict()
            self.pull_q_params_special_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_a_params, self.global_AC.special_a_params_dict[key])]}
                kv_c = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_c_params, self.global_AC.special_c_params_dict[key])]}
                self.pull_a_params_special_dict.update(kv_a)
                self.pull_c_params_special_dict.update(kv_c)


    def _prepare_many_goal_loss_and_update(self):
        self.many_goal_loss =  -tf.reduce_mean(tf.one_hot(self.a,4,dtype=tf.float32)*
                                tf.log(tf.clip_by_value(self.global_a_prob,1e-10,1.0)))
        self.many_goal_grad = [tf.clip_by_norm(item, 40) for item in
                               tf.gradients(self.many_goal_loss, self.global_a_params)]
        self.update_may_goals_op = self.OPT_A.apply_gradients(list(zip(self.many_goal_grad, self.global_AC.global_a_params)))


    def update_special(self, feed_dict,target_id):  # run by a local
        self.session.run([self.update_special_a_dict[target_id],
                          self.update_special_c_dict[target_id]],feed_dict)

    def update_global(self,feed_dict):
        self.session.run(self.update_global_a_op, feed_dict)  # local grads applies to global net

    def update_with_many_goals(self,current_state,next_state,action):
        self.session.run(self.update_may_goals_op,feed_dict = {self.s:current_state[np.newaxis, :],
                                                               self.t:next_state[np.newaxis, :],
                                                               self.a:np.array([action])})
        m_g_loss = self.session.run(self.many_goal_loss,feed_dict = {
                                                               self.s:current_state[np.newaxis, :],
                                                               self.t:next_state[np.newaxis, :],
                                                               self.a:np.array([action])})
        return m_g_loss


    def pull_global(self):
        self.session.run([self.pull_a_params_global])


    def pull_special(self,target_id):  # run by a local
        self.session.run([self.pull_a_params_special_dict[target_id]
                         ,self.pull_c_params_special_dict[target_id]])

    def spe_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.special_a_prob, feed_dict={self.s: s[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def glo_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.global_a_prob, feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights.ravel()


    def load_weight(self,target_id):
        self.session.run([self.pull_a_params_special_dict[target_id],self.pull_c_params_special_dict[target_id]])


    def get_special_value(self,feed_dict):
        special_value = self.session.run(self.special_v,feed_dict)
        return special_value


    def _prepare_store(self):
        var = tf.global_variables()
        var_to_restore = [val for val in var if 'Global_Net' in val.name ]
        self.saver = tf.train.Saver(var_to_restore )

    def store(self,weight_path):
        self.saver.save(self.session,weight_path)



