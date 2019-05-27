from utils.op import *
import tensorflow as tf
import numpy as np
from config.params import *
from config.constant import *

class ACNet(object):
    def __init__(self, scope,session,device,N_S,N_A,globalAC=None):
        tf.set_random_seed(50)
        with tf.device(device):
            self.dim_a = N_A
            self.dim_s = N_S
            self.session = session

            if scope == 'Global_Net':  # get global network
                with tf.variable_scope(scope):
                    self.s = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
                    self.t = tf.placeholder(tf.float32, [None, self.dim_s], 'Target')
                    self.T = tf.placeholder(tf.float32, None,'Temperature')

                    self.global_a_params,self.global_c_params = self._build_global_params_dict(scope)
                    self.special_a_params_dict,self.special_c_params_dict,self.special_q_params_dict =\
                        self._build_special_params_dict(scope)

            else:  # local net, calculate losses
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.s = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
                    self.a = tf.placeholder(tf.int32, [None, ], 'Action')
                    self.T = tf.placeholder(tf.float32,None,'Temperature')
                    self.global_v_target  = tf.placeholder(tf.float32, [None, 1], 'global_V_target')
                    self.special_v_target = tf.placeholder(tf.float32, [None, 1], 'special_V_target')
                    self.state_v_reg      = tf.placeholder(tf.float32, [None, 1], 'state_v_reg')
                    self.t = tf.placeholder(tf.float32, [None, self.dim_s], 'T')

                    self.OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
                    self.OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
                    self.OPT_REG_A = tf.train.RMSPropOptimizer(LR_REG_A, name='REG_RMSPropA')
                    self.OPT_REG_C = tf.train.RMSPropOptimizer(LR_REG_C, name='REG_RMSPropC')

                    self.global_a_prob ,self.global_v ,self.global_a_params ,self.global_c_params\
                        = self._build_global_net(scope)

                    self.special_a_prob,self.special_v,self.special_a_params,self.special_c_params,self.special_logits\
                        = self._build_special_net(scope)

                    self.q_value,self.special_q_params = self._build_q_net(scope)

                    self._prepare_global_loss(scope)

                    self._prepare_special_loss(scope)

                    self._prepare_global_grads(scope)

                    self._prepare_special_grads(scope)

                    self._prepare_update_op(scope)

                    self._prepare_pull_op(scope)

    def _build_global_params_dict(self, scope):
        with tf.variable_scope(scope):
            # encode
            w_encode = generate_fc_weight(shape=[self.dim_s, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512]        , name='global_b_encode')
            # fusion
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512]        , name='global_b_f')
            # scene
            w_scene = generate_fc_weight(shape=[512, 512]  , name='global_w_s')
            b_scene = generate_fc_bias(shape=[512]         , name='global_b_s')
            # actor
            w_actor = generate_fc_weight(shape=[512, self.dim_a] , name='global_w_a')
            b_actor = generate_fc_bias(shape=[self.dim_a]        , name='global_b_a')
            # critic
            w_critic = generate_fc_weight(shape=[512, 1]   , name='global_w_c')
            b_critic = generate_fc_bias(shape=[1]          , name='global_b_c')
            a_params = [w_encode, b_encode, w_fusion, b_fusion,w_scene, b_scene, w_actor,  b_actor]
            c_params = [w_encode, b_encode, w_fusion, b_fusion,w_scene, b_scene, w_critic, b_critic]
            return a_params, c_params

    def _build_special_params_dict(self,scope):
        with tf.variable_scope(scope):
            a_params_dict,c_params_dict = dict(),dict()
            q_params_dict = dict()
            for target_key in TARGET_ID_LIST:
                w_actor  = generate_fc_weight(shape=[self.dim_s, self.dim_a], name='actor_w'+str(target_key))
                b_actor  = generate_fc_bias(shape=[self.dim_a],               name='actor_b'+str(target_key))
                w_critic = generate_fc_weight(shape=[self.dim_s, 1],          name='critic_w'+str(target_key))
                b_critic = generate_fc_bias(shape=[1],                        name='critic_b'+str(target_key))
                w_q_net = generate_fc_weight(shape=[self.dim_s + self.dim_a, 1], name='w_q_net'+str(target_key))
                b_q_net = generate_fc_bias(shape=[1], name='b_q_net'+str(target_key))
                a_params = [w_actor  ,b_actor ]
                c_params = [w_critic ,b_critic]
                q_params = [w_q_net  ,b_q_net ]
                kv_a = {target_key:a_params}
                kv_c = {target_key:c_params}
                kv_q = {target_key:q_params}
                a_params_dict.update(kv_a)
                c_params_dict.update(kv_c)
                q_params_dict.update(kv_q)
            return  a_params_dict,c_params_dict,q_params_dict

    def _build_global_net(self, scope):
        with tf.variable_scope(scope):
            # encode
            w_encode = generate_fc_weight(shape=[self.dim_s, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512], name='global_b_encode')
            s_encode = tf.nn.elu(tf.matmul(self.s, w_encode) + b_encode)
            t_encode = tf.nn.elu(tf.matmul(self.t, w_encode) + b_encode)

            concat = tf.concat([s_encode, t_encode], axis=1)

            # fusion_layer
            w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[512], name='global_b_f')
            fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

            # scene_layer
            w_scene = generate_fc_weight(shape=[512, 512], name='global_w_s')
            b_scene = generate_fc_bias(shape=[512], name='global_b_s')
            scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

            # actor
            w_actor = generate_fc_weight(shape=[512, self.dim_a], name='global_w_a')
            b_actor = generate_fc_bias(shape=[self.dim_a], name='global_b_a')
            prob = tf.nn.softmax(tf.matmul(scene_layer, w_actor) + b_actor)

            # critic
            w_critic = generate_fc_weight(shape=[512, 1], name='global_w_c')
            b_critic = generate_fc_bias(shape=[1], name='global_b_c')
            value = tf.matmul(scene_layer, w_critic) + b_critic

            a_params = [w_encode, b_encode,w_fusion, b_fusion,
                        w_scene , b_scene ,w_actor , b_actor ]
            c_params = [w_encode, b_encode,w_fusion, b_fusion,
                        w_scene , b_scene ,w_critic, b_critic]

            return prob, value, a_params, c_params

    def _build_special_net(self, scope):
        with tf.variable_scope(scope):
            # actor
            w_actor = generate_fc_weight(shape=[self.dim_s, self.dim_a], name='special_w_a')
            b_actor = generate_fc_bias(shape=[self.dim_a], name='special_b_a')
            logits = tf.matmul(self.s, w_actor) + b_actor
            prob = tf.nn.softmax(logits)

            # critic
            w_critic = generate_fc_weight(shape=[self.dim_s, 1], name='special_w_c')
            b_critic = generate_fc_bias(shape=[1], name='special_b_c')
            value = tf.matmul(self.s, w_critic) + b_critic

            a_params = [w_actor,  b_actor ]
            c_params = [w_critic, b_critic]

            return prob, value, a_params, c_params,logits

    def _build_q_net(self,scope):
        with tf.variable_scope(scope):
            q_input = tf.concat([self.s,tf.one_hot(self.a,self.dim_a,dtype=tf.float32)],axis=1)
            w_q_net = generate_fc_weight(shape=[self.dim_s+self.dim_a,1], name='w_q_net')
            b_q_net = generate_fc_bias(shape=[1], name='b_q_net')
            q_value = tf.matmul(q_input,w_q_net)+b_q_net
            q_params = [w_q_net,b_q_net]
            return q_value,q_params



    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):
            self.glo_td = tf.subtract(self.global_v_target,self.global_v,name = 'glo_TD_error')
            self.reg_td = tf.subtract(self.state_v_reg    ,self.global_v,name='reg_TD_error')
            with tf.name_scope('global_c_loss'):
                reg_c_loss = tf.reduce_mean(tf.square(self.reg_td))
                glo_c_loss = tf.reduce_mean(tf.square(self.glo_td))
                #self.global_c_loss = 0.8*reg_c_loss + 0.2*glo_c_loss
                #self.global_c_loss = BETA_REG_VALUE*(0.5*reg_c_loss + 0.5*glo_c_loss)
                self.global_c_loss = reg_c_loss


            with tf.name_scope('global_a_loss'):
                p_target = tf.stop_gradient(tf.nn.softmax(self.special_logits*self.T))
                p_update = self.global_a_prob
                distribution_error = -tf.reduce_mean(p_target*tf.log(tf.clip_by_value(p_update,1e-10,1.0)))

                glo_log_prob = tf.reduce_sum(tf.log(self.global_a_prob + 1e-5)*tf.one_hot(self.a, self.dim_a, dtype=tf.float32),
                                             axis=1,keep_dims=True)
                glo_exp_v = glo_log_prob * tf.stop_gradient(self.glo_td)

                self.glo_entropy = -tf.reduce_mean(self.global_a_prob*tf.log(self.global_a_prob + 1e-5), axis=1,keep_dims=True)  # encourage exploration
                # self.exp_v = ENTROPY_BETA*self.glo_entropy + BETA_REG_ACTION*distribution_error + glo_exp_v
                self.exp_v = ENTROPY_BETA*self.glo_entropy + glo_exp_v

                self.kl = self.KL_divergence(p_stable=p_target, p_advance=p_update)
                self.kl_mean = tf.reduce_mean(self.kl)
                self.global_a_loss = tf.reduce_mean(-self.exp_v)


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

            with tf.name_scope('special_q_loss'):
                self.q_error = tf.subtract(self.q_value,self.special_v_target)
                self.special_q_loss = tf.reduce_mean(tf.square(self.q_error))


    def _prepare_global_grads(self,scope):
        with tf.name_scope(scope+'global_grads'):
            with tf.name_scope('global_net_grad'):
                self.global_a_grads = [tf.clip_by_norm(item, 40) for item in
                                       tf.gradients(self.global_a_loss, self.global_a_params)]

                self.global_c_grads = [tf.clip_by_norm(item, 40) for item in
                                       tf.gradients(self.global_c_loss, self.global_c_params)]

    def _prepare_special_grads(self,scope):
        with tf.name_scope(scope+'special_grads'):
            with tf.name_scope('special_net_grad'):
                self.special_a_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.special_a_loss, self.special_a_params)]

                self.special_c_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.special_c_loss, self.special_c_params)]

                self.special_q_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.special_q_loss, self.special_q_params)]

    def _prepare_update_op(self,scope):
        with tf.name_scope(scope+'_global_update'):
            self.update_global_a_op = self.OPT_A.apply_gradients(list(zip(self.global_a_grads, self.global_AC.global_a_params)))
            self.update_global_c_op = self.OPT_C.apply_gradients(list(zip(self.global_c_grads, self.global_AC.global_c_params)))
        with tf.name_scope(scope+'_special_update'):
            self.update_special_a_dict, self.update_special_c_dict = dict(), dict()
            self.update_special_q_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: self.OPT_REG_A.apply_gradients(list(zip(self.special_a_grads , self.global_AC.special_a_params_dict[key])))}
                kv_c = {key: self.OPT_REG_C.apply_gradients(list(zip(self.special_c_grads , self.global_AC.special_c_params_dict[key])))}
                kv_q = {key: self.OPT_REG_C.apply_gradients(list(zip(self.special_q_grads , self.global_AC.special_q_params_dict[key])))}
                self.update_special_a_dict.update(kv_a)
                self.update_special_c_dict.update(kv_c)
                self.update_special_q_dict.update(kv_q)

    def _prepare_pull_op(self,scope):
        with tf.name_scope(scope+'pull_global_params'):
            self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in
                                         zip(self.global_a_params, self.global_AC.global_a_params)]
            self.pull_c_params_global = [l_p.assign(g_p) for l_p, g_p in
                                         zip(self.global_c_params, self.global_AC.global_c_params)]

        with tf.name_scope(scope+'pull_special_params'):
            self.pull_a_params_special_dict, self.pull_c_params_special_dict = dict(), dict()
            self.pull_q_params_special_dict = dict()
            for key in TARGET_ID_LIST:
                kv_a = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_a_params, self.global_AC.special_a_params_dict[key])]}
                kv_c = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_c_params, self.global_AC.special_c_params_dict[key])]}
                kv_q = {key: [l_p.assign(g_p) for l_p, g_p in
                              zip(self.special_q_params, self.global_AC.special_q_params_dict[key])]}
                self.pull_a_params_special_dict.update(kv_a)
                self.pull_c_params_special_dict.update(kv_c)
                self.pull_q_params_special_dict.update(kv_q)


    def compute_kl(self,feed_dict):
        kl = self.session.run(self.kl_mean,feed_dict=feed_dict)
        return kl

    def update_special(self, feed_dict,target_id):  # run by a local
        self.session.run([self.update_special_a_dict[target_id],
                          self.update_special_c_dict[target_id],
                          self.update_special_q_dict[target_id]],feed_dict)

    def update_global(self,feed_dict):
        self.session.run([self.update_global_a_op, self.update_global_c_op], feed_dict)  # local grads applies to global net


    def pull_global(self):
        self.session.run([self.pull_a_params_global, self.pull_c_params_global])


    def pull_special(self,target_id):  # run by a local
        self.session.run([self.pull_a_params_special_dict[target_id]
                         ,self.pull_c_params_special_dict[target_id]
                         ,self.pull_q_params_special_dict[target_id]])

    def spe_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.special_a_prob, feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def glo_choose_action(self, s, t):  # run by a local
        prob_weights = self.session.run(self.global_a_prob, feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights.ravel()


    def load_weight(self,target_id):
        self.session.run([self.pull_a_params_special_dict[target_id],
                          self.pull_c_params_special_dict[target_id],
                          self.pull_q_params_special_dict[target_id]])

    def KL_divergence(self,p_stable,p_advance):
        X = tf.distributions.Categorical(probs = p_stable )
        Y = tf.distributions.Categorical(probs = p_advance)
        return tf.clip_by_value(tf.distributions.kl_divergence(X, Y), clip_value_min=0.0, clip_value_max=10)

    def get_q_value(self,s,a):
        s,a = np.array([s]),np.array([a])
        q = self.session.run(self.q_value,feed_dict={self.s:s,self.a:a})[0,0]
        return q
